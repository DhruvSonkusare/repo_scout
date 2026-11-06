"""The only place chat and embedding models are created (gemini or fake)."""

import asyncio
import contextlib
import logging
import math
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, time as clock_time, timedelta
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Literal, NoReturn
from zoneinfo import ZoneInfo

import httpx
from google.genai import Client
from google.genai.errors import APIError
from google.genai.types import HttpOptions, HttpRetryOptions
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel, FakeListChatModel, LanguageModelInput
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, convert_to_messages
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from pydantic import BaseModel

from app.config import Settings, ThinkingLevel, get_settings

# FastEmbed and ONNX Runtime take a lot of memory, and the deployed app doesn't install them.
# So they're only imported by the functions that use local embeddings.
if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

FAKE_CHAT_RESPONSES = ["This is a fake answer from the offline test model [1]."]
FAKE_MODEL_NAME = "fake-model"
FAKE_EMBEDDING_SIZE = 256
# FastEmbed's one model trained specifically on source code.
DEFAULT_LOCAL_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
LOCAL_EMBED_BATCH_SIZE = 256

# The only errors that move on to the next model: a model that's gone, rate limits, quotas, overload
# and server errors. A 400 means our request is wrong, which the next model can't fix.
MODEL_NOT_FOUND_CODE = 404
FALLBACK_STATUS_CODES = {MODEL_NOT_FOUND_CODE, 429, 500, 503, 504}
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
# Google resets daily quotas at midnight Pacific time.
QUOTA_RESET_ZONE = ZoneInfo("America/Los_Angeles")
# Gemini 2.5 takes a token budget instead of a level. 0 turns thinking off.
THINKING_BUDGETS = {"minimal": 0, "low": 1024, "medium": 8192, "high": 24576}
MODEL_LIST_TIMEOUT_MS = 10_000
CLIENT_CLOSE_TIMEOUT_SECONDS = 5.0


class ModelUnavailableError(Exception):
    """A model is rate limited, out of quota or overloaded, so the next one in the chain should answer."""


class ModelSetupError(ValueError):
    """The .env settings can't produce a working model. The message says what to change."""


# Everything a Gemini call can raise. Chat and embedding errors wrap an APIError in __cause__.
GEMINI_ERRORS = (APIError, GoogleGenerativeAIError, httpx.HTTPError, ModelUnavailableError)
GEMINI_ERROR_MESSAGES = {
    400: "Gemini rejected the request. Check GOOGLE_API_KEY and the model names in .env.",
    401: "Gemini rejected the API key. Check GOOGLE_API_KEY in .env.",
    403: "This API key isn't allowed to use that model. Check GOOGLE_API_KEY in .env.",
    404: "Gemini can't find that model. Run scripts/list_models.py and check the names in .env.",
    429: "Gemini's rate limit or quota was reached. Wait a minute and try again.",
}
GEMINI_BUSY_MESSAGE = "Gemini is busy right now. Try again shortly."
GEMINI_DAILY_QUOTA_MESSAGE = "Gemini's daily quota for this model is used up. Use a different model, or try again tomorrow."
ALL_MODELS_UNAVAILABLE_MESSAGE = (
    "Every configured model is unavailable: busy, rate limited or out of daily quota. Try again in a minute, or "
    "tomorrow if the daily quotas are used up. The server log has the details for each model."
)


def _api_error(error: Exception) -> APIError | None:
    """Return the Gemini API error behind an exception, if there is one."""
    return error if isinstance(error, APIError) else error.__cause__ if isinstance(error.__cause__, APIError) else None


def _error_details(api_error: APIError) -> list[dict[str, Any]]:
    """Return the structured details from a Gemini error response, such as RetryInfo."""
    body = api_error.details
    error = body.get("error") if isinstance(body, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    return [item for item in details if isinstance(item, dict)] if isinstance(details, list) else []


def is_daily_quota_error(error: Exception) -> bool:
    """Return True for a per-day quota, which waiting a few minutes can't fix."""
    api_error = _api_error(error)
    if api_error is None or api_error.code != 429:
        return False
    return any(
        "PerDay" in str(violation.get("quotaId", ""))
        for detail in _error_details(api_error)
        for violation in detail.get("violations", [])
        if isinstance(violation, dict)
    )


def gemini_retry_delay(error: Exception, default_seconds: float) -> float | None:
    """Return how long to wait before retrying, or None if retrying won't help.

    Gemini's 429 responses can include a RetryInfo detail such as `"retryDelay": "37s"`. The
    Google SDK's own retries ignore it, so callers doing long batch work should use this value.
    """
    if is_daily_quota_error(error):
        return None
    api_error = _api_error(error)
    if api_error is None:
        is_network_error = isinstance(error, httpx.HTTPError) or isinstance(error.__cause__, httpx.HTTPError)
        return default_seconds if is_network_error else None
    if api_error.code >= 500:
        return default_seconds
    if api_error.code != 429:
        return None
    for detail in _error_details(api_error):
        if "RetryInfo" in str(detail.get("@type", "")):
            try:
                return float(str(detail.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                break
    return default_seconds


def describe_gemini_error(error: Exception) -> str:
    """Turn a Gemini or network error into a short message."""
    if isinstance(error, ModelUnavailableError):
        return ALL_MODELS_UNAVAILABLE_MESSAGE
    api_error = _api_error(error)
    if is_daily_quota_error(error):
        return f"{GEMINI_DAILY_QUOTA_MESSAGE} (HTTP {api_error.code})"
    if api_error is not None:
        if api_error.code >= 500:
            return f"{GEMINI_BUSY_MESSAGE} (HTTP {api_error.code})"
        known = GEMINI_ERROR_MESSAGES.get(api_error.code, "Gemini returned an error.")
        return f"{known} (HTTP {api_error.code})"
    if isinstance(error, httpx.HTTPError) or isinstance(error.__cause__, httpx.HTTPError):
        return "Couldn't reach Gemini. Check your internet connection."
    return "Gemini returned an unexpected error."


def _require_gemini_setting(value: str, env_name: str) -> str:
    """Return a Gemini setting, or explain which .env value is missing."""
    if not value:
        raise ModelSetupError(f"{env_name} is not set. Add it to .env, or set LLM_PROVIDER=fake to run offline.")
    return value


def _gemini_api_key(settings: Settings) -> str:
    """Return the Gemini API key, or explain that it is missing."""
    return _require_gemini_setting(settings.google_api_key.get_secret_value(), "GOOGLE_API_KEY")


@dataclass(frozen=True)
class ModelProfile:
    """What one model family accepts."""

    thinking: Literal["level", "budget"] | None = None
    system_prompt: bool = True
    structured_output: bool = True


def model_profile(name: str) -> ModelProfile:
    """Return what a model accepts, going by its name. A family we don't know gets only the basics."""
    short_name = name.removeprefix("models/")
    if short_name.startswith("gemini-3"):
        return ModelProfile(thinking="level")
    if short_name.startswith("gemini-2.5"):
        return ModelProfile(thinking="budget")
    if short_name.startswith("gemini-") or short_name == FAKE_MODEL_NAME:
        return ModelProfile()
    if short_name.startswith("gemma-"):
        return ModelProfile(system_prompt=False, structured_output=False)
    return ModelProfile(structured_output=False)


def _thinking_settings(profile: ModelProfile, level: ThinkingLevel) -> dict[str, Any]:
    """Return the thinking option this model family understands, if it has one.

    Gemini models "think" before writing, and the user sees nothing until that ends. Answers here
    come from excerpts we supply, so they need little of it, and less means a faster first token.
    """
    if level == "default" or profile.thinking is None:
        return {}
    if profile.thinking == "level":
        return {"thinking_level": level}
    return {"thinking_budget": THINKING_BUDGETS[level]}


def _merge_system_prompt(model_input: LanguageModelInput) -> LanguageModelInput:
    """Move system messages into the first user message, for models without system instructions."""
    if isinstance(model_input, str):
        return model_input
    messages = convert_to_messages(model_input)
    system_prompt = "\n\n".join(message.text for message in messages if isinstance(message, SystemMessage))
    others: list[BaseMessage] = [message for message in messages if not isinstance(message, SystemMessage)]
    if not system_prompt:
        return others
    for index, message in enumerate(others):
        if isinstance(message, HumanMessage):
            others[index] = HumanMessage(f"{system_prompt}\n\n{message.text}")
            return others
    return [HumanMessage(system_prompt), *others]


class CircuitBreaker:
    """Remembers, in memory, which models to skip until when, so we don't waste calls on them."""

    def __init__(self) -> None:
        self._open_until: dict[str, float] = {}

    def is_open(self, name: str) -> bool:
        """Return True while this model should be skipped."""
        return _now() < self._open_until.get(name, 0.0)

    def open_until(self, name: str, until: float) -> None:
        """Skip this model until the given time."""
        self._open_until[name] = until

    def disable(self, name: str) -> None:
        """Skip this model until the server restarts."""
        self._open_until[name] = math.inf

    def is_disabled(self, name: str) -> bool:
        """Return True if this model is skipped until the server restarts."""
        return self._open_until.get(name) == math.inf

    def reset(self) -> None:
        """Forget every model."""
        self._open_until.clear()


circuit_breaker = CircuitBreaker()  # shared, because chat and guide chains use the same quotas


def _now() -> float:
    """Return the current time. Wall-clock time, because a quota reset is a time of day."""
    return time.time()


def next_quota_reset(now: float) -> float:
    """Return the next midnight Pacific time, when daily quotas reset."""
    tomorrow = datetime.fromtimestamp(now, QUOTA_RESET_ZONE).date() + timedelta(days=1)
    return datetime.combine(tomorrow, clock_time.min, tzinfo=QUOTA_RESET_ZONE).timestamp()


def _is_answer_text(chunk: Any) -> bool:
    """Return True for a chunk with text to show. Structured outputs aren't messages, and always count."""
    return not isinstance(chunk, BaseMessage) or bool(chunk.text)


class _ChainLink(Runnable[LanguageModelInput, Any]):
    """One model in a fallback chain, which decides whether a failure should move on to the next model.

    with_fallbacks can only filter by exception class, and a 429 and a 400 are the same class. So
    this checks the status code, raising ModelUnavailableError only when falling back is right.
    The model is built on its turn, because creating each Google client takes about half a second.
    """

    def __init__(self, name: str, build: Callable[[], Runnable], on_answer: Callable[[str], None] | None) -> None:
        self.name = name
        self.build = build
        self.on_answer = on_answer

    def _start(self, model_input: LanguageModelInput) -> tuple[Runnable, LanguageModelInput]:
        """Return the model and its input, unless the circuit breaker says to skip this model."""
        if circuit_breaker.is_open(self.name):
            logger.info("Skipping model %s until its rate limit or quota resets.", self.name)
            raise ModelUnavailableError(f"{self.name} is skipped by the circuit breaker")
        accepts_system_prompt = model_profile(self.name).system_prompt
        return self.build(), model_input if accepts_system_prompt else _merge_system_prompt(model_input)

    def _answered(self) -> None:
        """Report that this model is the one answering."""
        logger.info("Model %s answered.", self.name)
        if self.on_answer is not None:
            self.on_answer(self.name)

    def _handle_failure(self, error: Exception, answer_started: bool = False) -> NoReturn:
        """Update the circuit breaker, then move on to the next model if that's allowed."""
        api_error = _api_error(error)
        code = api_error.code if api_error is not None else None
        if code == MODEL_NOT_FOUND_CODE:
            circuit_breaker.disable(self.name)
            logger.error("Model %s was not found or is unavailable (HTTP 404). Disabled until restart.", self.name)
        elif is_daily_quota_error(error):
            circuit_breaker.open_until(self.name, next_quota_reset(_now()))
        elif code == 429:
            circuit_breaker.open_until(self.name, _now() + gemini_retry_delay(error, RATE_LIMIT_COOLDOWN_SECONDS))
        if code in FALLBACK_STATUS_CODES and not answer_started:
            logger.warning("Model %s is unavailable (HTTP %s), so trying the next one.", self.name, code)
            raise ModelUnavailableError(f"{self.name} returned HTTP {code}") from error
        logger.warning("Model %s failed with no fallback (HTTP %s).", self.name, code)
        raise error

    def invoke(self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Return this model's whole reply."""
        model, model_input = self._start(input)
        try:
            result = model.invoke(model_input, config, **kwargs)
        except GEMINI_ERRORS as error:
            self._handle_failure(error)
        self._answered()
        return result

    async def ainvoke(self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Return this model's whole reply, without blocking the event loop."""
        model, model_input = self._start(input)
        try:
            result = await model.ainvoke(model_input, config, **kwargs)
        except GEMINI_ERRORS as error:
            self._handle_failure(error)
        self._answered()
        return result

    async def astream(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Yield this model's reply piece by piece.

        with_fallbacks falls back only until a link yields its first chunk. Empty chunks are held
        back, so a model that fails after sending one can still fall back: the user has seen nothing.
        """
        model, model_input = self._start(input)
        answer_started = False
        try:
            async for chunk in model.astream(model_input, config, **kwargs):
                if not answer_started and _is_answer_text(chunk):
                    answer_started = True
                    self._answered()
                if answer_started:
                    yield chunk
        except GEMINI_ERRORS as error:
            self._handle_failure(error, answer_started)
        if not answer_started:
            self._answered()
            yield AIMessageChunk(content="")


# Each model holds a Google client with open connections, so keep one per set of options instead of
# one per request. That saves about 0.5 s per client and some memory, and leaves a known list to
# close at shutdown.
_gemini_models: dict[tuple[str, int, ThinkingLevel], ChatGoogleGenerativeAI] = {}
_gemini_embeddings: dict[tuple[str, int], GoogleGenerativeAIEmbeddings] = {}


def _fake_chat_model(name: str) -> BaseChatModel:
    """Return the offline model. FakeListChatModel replies with these strings and never touches the network."""
    return FakeListChatModel(responses=FAKE_CHAT_RESPONSES)


def _build_chat_model(name: str, is_last: bool, thinking_level: ThinkingLevel) -> BaseChatModel:
    """Create one model of a chain, with the options its family accepts."""
    settings = get_settings()
    if settings.llm_provider == "fake":
        return _fake_chat_model(name)
    # The next model is a faster retry than waiting out a backoff, so only the last one retries.
    retries = settings.max_retries if is_last else 0
    key = (name, retries, thinking_level)
    if key not in _gemini_models:
        _gemini_models[key] = ChatGoogleGenerativeAI(
            model=name,
            google_api_key=_gemini_api_key(settings),
            temperature=0.2,
            # Despite its name, this counts every attempt including the first, and 0 means "use
            # Google's default", so add 1 to make MAX_RETRIES=0 really mean no retries.
            max_retries=retries + 1,
            **_thinking_settings(model_profile(name), thinking_level),
        )
    return _gemini_models[key]


async def close_models() -> None:
    """Close every Google client, for server shutdown. A failure or a slow close can't block the exit."""
    for embeddings in _gemini_embeddings.values():
        embeddings.client.close()
    _gemini_embeddings.clear()
    models = list(_gemini_models.values())
    _gemini_models.clear()
    for model in models:
        try:
            await asyncio.wait_for(model.aclose(), CLIENT_CLOSE_TIMEOUT_SECONDS)
        except (OSError, RuntimeError, TimeoutError) as error:
            logger.warning("Couldn't close the client of model %s: %r", model.model, error)


# Names that startup didn't find in Gemini's model list, which are left out of every chain.
_unknown_models: set[str] = set()


def _configured_chains(settings: Settings) -> dict[str, list[str]]:
    """Return each chain's model names as written in .env, by setting name."""
    return {"CHAT_MODELS": settings.chat_models, "GUIDE_MODELS": settings.guide_models}


def _list_generation_models(api_key: str) -> set[str]:
    """Return the names of the models this API key can generate with. Listing uses no generation quota."""
    client = Client(
        api_key=api_key,
        http_options=HttpOptions(timeout=MODEL_LIST_TIMEOUT_MS, retry_options=HttpRetryOptions(attempts=1)),
    )
    with contextlib.closing(client):
        return {
            (model.name or "").removeprefix("models/")
            for model in client.models.list(config={"page_size": 1000})
            if "generateContent" in (model.supported_actions or [])
        }


def validate_model_chains() -> None:
    """Check CHAT_MODELS and GUIDE_MODELS against Gemini's model list, and drop names it doesn't have.

    A wrong name returns a 404, which never falls back, so it would stop every request that
    reaches it. If the list can't be fetched, every model stays: a network blip at startup
    shouldn't empty a working chain.
    """
    settings = get_settings()
    _unknown_models.clear()
    api_key = settings.google_api_key.get_secret_value()
    if settings.llm_provider == "fake" or not api_key:
        return
    try:
        available = _list_generation_models(api_key)
    except GEMINI_ERRORS as error:
        logger.warning("Couldn't check the model names, so keeping them all: %s", describe_gemini_error(error))
        return
    for env_name, names in _configured_chains(settings).items():
        for name in names:
            if name.removeprefix("models/") not in available:
                _unknown_models.add(name)
                logger.warning(
                    "%s: dropping %r, because Gemini's model list has no such model for generating text. "
                    "Run scripts/list_models.py to see the names.",
                    env_name,
                    name,
                )
        if names and set(names) <= _unknown_models:
            logger.error("%s has no usable models left. /api/health will report this.", env_name)


def _chain_names(names: list[str], env_name: str) -> list[str]:
    """Return a chain's usable model names, or explain what to fix in .env."""
    settings = get_settings()
    if settings.llm_provider == "fake":
        return names or [FAKE_MODEL_NAME]
    _gemini_api_key(settings)
    _require_gemini_setting(",".join(names), env_name)
    usable = [name for name in names if name not in _unknown_models and not circuit_breaker.is_disabled(name)]
    if not usable:
        raise ModelSetupError(
            f"{env_name} has no usable models: Gemini doesn't offer any of {', '.join(names)}. "
            "Run scripts/list_models.py, fix the names in .env and restart the server."
        )
    return usable


def model_chain_problem() -> str | None:
    """Explain why a chain can't run, for the health check, or return None if they all can."""
    settings = get_settings()
    problems: list[str] = []
    for env_name, names in _configured_chains(settings).items():
        try:
            _chain_names(names, env_name)
        except ModelSetupError as error:
            problems.append(str(error))
    return " ".join(dict.fromkeys(problems)) or None


def _model_chain(
    names: list[str],
    build: Callable[[str, bool], Runnable],
    on_answer: Callable[[str], None] | None = None,
) -> Runnable:
    """Join models into one runnable that tries each in order.

    with_fallbacks returns a runnable that works like the first model, but on one of the listed
    exceptions tries the next. When streaming, it can only switch before the first chunk, since
    text already sent can't be taken back. Chains use it like any model: `prompt | model | parser`.
    """
    links = [
        _ChainLink(name, partial(build, name, index == len(names) - 1), on_answer) for index, name in enumerate(names)
    ]
    return links[0].with_fallbacks(links[1:], exceptions_to_handle=(ModelUnavailableError,))


def get_chat_model(on_answer: Callable[[str], None] | None = None) -> Runnable:
    """Return the CHAT_MODELS chain, for answers. on_answer receives the name of the model that answers."""
    settings = get_settings()
    build = partial(_build_chat_model, thinking_level=settings.chat_thinking_level)
    return _model_chain(_chain_names(settings.chat_models, "CHAT_MODELS"), build, on_answer)


def get_guide_model() -> Runnable:
    """Return the GUIDE_MODELS chain, for long documents, which get their own thinking level."""
    settings = get_settings()
    build = partial(_build_chat_model, thinking_level=settings.guide_thinking_level)
    return _model_chain(_chain_names(settings.guide_models, "GUIDE_MODELS"), build)


def _structured_chain(
    names: list[str],
    env_name: str,
    thinking_level: ThinkingLevel,
    schema: type[BaseModel],
    fake_reply: Callable[[Any], BaseModel] | None,
) -> Runnable:
    """Chain the models that can reply in a schema's shape.

    with_structured_output makes a model reply in the schema's shape and parses the reply into the
    class. Each model needs its own schema request, so it's applied to each model and then those
    are chained, leaving out models that can't do it. The fake provider replies with fake_reply.
    """
    capable = [name for name in names if model_profile(name).structured_output]
    if not capable:
        raise ModelSetupError(f"No model in {env_name} supports structured output. Add a Gemini model to it.")

    def build(name: str, is_last: bool) -> Runnable:
        if get_settings().llm_provider == "fake" and fake_reply is not None:
            return RunnableLambda(fake_reply)
        return _build_chat_model(name, is_last, thinking_level).with_structured_output(schema)

    return _model_chain(capable, build)


def get_structured_model(schema: type[BaseModel], fake_reply: Callable[[Any], BaseModel] | None = None) -> Runnable:
    """Return the CHAT_MODELS chain, replying with instances of a Pydantic class."""
    settings = get_settings()
    names = _chain_names(settings.chat_models, "CHAT_MODELS")
    return _structured_chain(names, "CHAT_MODELS", settings.chat_thinking_level, schema, fake_reply)


def get_structured_guide_model(
    schema: type[BaseModel], fake_reply: Callable[[Any], BaseModel] | None = None
) -> Runnable:
    """Return the GUIDE_MODELS chain, replying with instances of a Pydantic class, for work on long documents."""
    settings = get_settings()
    names = _chain_names(settings.guide_models, "GUIDE_MODELS")
    return _structured_chain(names, "GUIDE_MODELS", settings.guide_thinking_level, schema, fake_reply)


def embedding_identity() -> tuple[str, str]:
    """Return the (provider, model) in use now. Vectors only match others from the same pair."""
    settings = get_settings()
    if settings.embedding_provider == "fake":
        return "fake", f"deterministic-{FAKE_EMBEDDING_SIZE}"
    if settings.embedding_provider == "local":
        return "local", settings.embedding_model or DEFAULT_LOCAL_EMBEDDING_MODEL
    return "gemini", settings.embedding_model


class CudaUnavailableError(RuntimeError):
    """CUDA can't be used for local embeddings. The message says why."""


def _load_on_cuda(model_name: str, cache_dir: str) -> "TextEmbedding":
    """Load a model on the GPU, or raise CudaUnavailableError.

    ONNX Runtime falls back to the CPU silently if CUDA fails to start, so this checks which
    provider the loaded session really uses.
    """
    import onnxruntime as ort
    from fastembed import TextEmbedding

    # On Windows, this finds the CUDA DLLs that requirements-gpu.txt installs through pip.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    if CUDA_PROVIDER not in ort.get_available_providers():
        raise CudaUnavailableError("this install has no CUDA support (see requirements-gpu.txt)")
    try:
        text_embedding = TextEmbedding(model_name, cache_dir=cache_dir, providers=[CUDA_PROVIDER, CPU_PROVIDER])
    except Exception as error:
        raise CudaUnavailableError(f"the model couldn't start on CUDA: {error}") from error
    if text_embedding.model.model.get_providers()[0] != CUDA_PROVIDER:
        raise CudaUnavailableError("ONNX Runtime couldn't start CUDA, so check the NVIDIA driver")
    return text_embedding


def _load_text_embedding(model_name: str, cache_dir: str, device: str) -> "TextEmbedding":
    """Load a FastEmbed model on the requested device, and log which one it really uses."""
    from fastembed import TextEmbedding

    if device != "cpu":
        try:
            text_embedding = _load_on_cuda(model_name, cache_dir)
            logger.info("Local embeddings: %s is running on CUDA.", model_name)
            return text_embedding
        except CudaUnavailableError as error:
            if device == "cuda":
                raise ValueError(f"EMBEDDING_DEVICE=cuda, but CUDA isn't usable: {error}.") from error
            logger.warning("Local embeddings: CUDA isn't usable (%s), so falling back to the CPU.", error)
    text_embedding = TextEmbedding(model_name, cache_dir=cache_dir, providers=[CPU_PROVIDER])
    logger.info("Local embeddings: %s is running on the CPU.", model_name)
    return text_embedding


class LocalEmbeddings(Embeddings):
    """A FastEmbed model behind LangChain's Embeddings interface, so vector stores can use it.

    The interface has two methods because many retrieval models treat the two sides differently:
    embed_documents is for the text being stored, and embed_query is for the search that finds it.
    Always using the right one means a model that does treat them differently still works here.
    """

    def __init__(self, model_name: str, model: "TextEmbedding") -> None:
        self.model_name = model_name
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per chunk of text being indexed."""
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=LOCAL_EMBED_BATCH_SIZE)]

    def embed_query(self, text: str) -> list[float]:
        """Return the vector for a search."""
        return next(iter(self.model.query_embed(text))).tolist()


@lru_cache
def _local_embeddings(model_name: str, cache_dir: str, device: str) -> LocalEmbeddings:
    """Load a FastEmbed model once. Loading takes seconds, so every search can't repeat it.

    FastEmbed runs the model on this machine with ONNX Runtime, so there's no API key, quota or
    per-request cost. The first use downloads the model into cache_dir.
    """
    from fastembed import TextEmbedding

    supported_models = {model["model"].lower() for model in TextEmbedding.list_supported_models()}
    if model_name.lower() not in supported_models:
        raise ValueError(
            f"EMBEDDING_MODEL {model_name!r} isn't a FastEmbed model. Change it in .env, or leave it empty "
            f"to use {DEFAULT_LOCAL_EMBEDDING_MODEL}."
        )
    return LocalEmbeddings(model_name, _load_text_embedding(model_name, cache_dir, device))


def get_embeddings() -> Embeddings:
    """Return the embedding model chosen by EMBEDDING_PROVIDER.

    Embeddings turn text into a list of numbers, so a vector store can find similar text by
    comparing the numbers. A vector store must use one model for both indexing and searching,
    because vectors from different models can't be compared.
    """
    settings = get_settings()
    if settings.embedding_provider == "fake":
        # The same text always gets the same vector, but the numbers don't capture meaning.
        return DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE)
    if settings.embedding_provider == "local":
        return _local_embeddings(embedding_identity()[1], str(settings.fastembed_dir), settings.embedding_device)
    api_key = _gemini_api_key(settings)
    model = _require_gemini_setting(settings.embedding_model, "EMBEDDING_MODEL")
    key = (model, settings.max_retries)
    if key not in _gemini_embeddings:
        embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        # This class has no max_retries and silently ignores the argument, so it would never retry.
        # Giving it a Google client configured with retries adds exponential backoff on 429 and 503.
        embeddings.client = Client(
            api_key=api_key,
            http_options=HttpOptions(retry_options=HttpRetryOptions(attempts=settings.max_retries + 1)),
        )
        _gemini_embeddings[key] = embeddings
    return _gemini_embeddings[key]
