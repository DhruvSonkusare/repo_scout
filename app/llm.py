"""The only place chat and embedding models are created (gemini or fake)."""

from functools import lru_cache
from typing import Any

import httpx
from google.genai import Client
from google.genai.errors import APIError
from google.genai.types import HttpOptions, HttpRetryOptions
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

from app.config import Settings, get_settings

FAKE_CHAT_RESPONSES = ["This is a fake answer from the offline test model [1]."]
FAKE_EMBEDDING_SIZE = 256
# FastEmbed's one model trained specifically on source code.
DEFAULT_LOCAL_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"

# Everything a Gemini call can raise. Chat errors are APIErrors; embedding errors wrap one in __cause__.
GEMINI_ERRORS = (APIError, GoogleGenerativeAIError, httpx.HTTPError)
GEMINI_ERROR_MESSAGES = {
    400: "Gemini rejected the request. Check GOOGLE_API_KEY and the model names in .env.",
    401: "Gemini rejected the API key. Check GOOGLE_API_KEY in .env.",
    403: "This API key isn't allowed to use that model. Check GOOGLE_API_KEY in .env.",
    404: "Gemini can't find that model. Run scripts/list_models.py and check the names in .env.",
    429: "Gemini's rate limit or quota was reached. Wait a minute and try again.",
}
GEMINI_BUSY_MESSAGE = "Gemini is busy right now. Try again shortly, or use a different CHAT_MODEL."


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
    api_error = _api_error(error)
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
        raise ValueError(f"{env_name} is not set. Add it to .env, or set LLM_PROVIDER=fake to run offline.")
    return value


def _gemini_api_key(settings: Settings) -> str:
    """Return the Gemini API key, or explain that it is missing."""
    return _require_gemini_setting(settings.google_api_key.get_secret_value(), "GOOGLE_API_KEY")


def get_chat_model() -> BaseChatModel:
    """Return the configured chat model.

    Every LangChain chat model shares the BaseChatModel interface (invoke, stream, and use in
    LCEL pipes like `prompt | model | parser`). So chains never need to know which provider they
    got, and swapping Gemini for the fake only changes this function.
    """
    settings = get_settings()
    if settings.llm_provider == "fake":
        # FakeListChatModel replies with these strings in turn and never touches the network.
        return FakeListChatModel(responses=FAKE_CHAT_RESPONSES)
    return ChatGoogleGenerativeAI(
        model=_require_gemini_setting(settings.chat_model, "CHAT_MODEL"),
        google_api_key=_gemini_api_key(settings),
        temperature=0.2,
        # Despite its name, this counts every attempt including the first, and 0 means "use
        # Google's default", so add 1 to make MAX_RETRIES=0 really mean no retries.
        max_retries=settings.max_retries + 1,
    )


@lru_cache
def _local_embeddings(model_name: str, cache_dir: str) -> FastEmbedEmbeddings:
    """Load a FastEmbed model once. Loading takes seconds, so every search can't repeat it.

    FastEmbed runs the model on this machine's CPU with ONNX Runtime, so there's no API key, quota
    or per-request cost. The first use downloads the model into cache_dir.
    """
    try:
        return FastEmbedEmbeddings(model_name=model_name, cache_dir=cache_dir)
    except ValueError as error:
        raise ValueError(
            f"EMBEDDING_MODEL {model_name!r} isn't a FastEmbed model. Change it in .env, or leave it empty "
            f"to use {DEFAULT_LOCAL_EMBEDDING_MODEL}."
        ) from error


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
        model_name = settings.embedding_model or DEFAULT_LOCAL_EMBEDDING_MODEL
        return _local_embeddings(model_name, str(settings.fastembed_dir))
    api_key = _gemini_api_key(settings)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=_require_gemini_setting(settings.embedding_model, "EMBEDDING_MODEL"),
        google_api_key=api_key,
    )
    # This class has no max_retries and silently ignores the argument, so it would never retry.
    # Giving it a Google client configured with retries adds exponential backoff on 429 and 503.
    embeddings.client = Client(
        api_key=api_key,
        http_options=HttpOptions(retry_options=HttpRetryOptions(attempts=settings.max_retries + 1)),
    )
    return embeddings
