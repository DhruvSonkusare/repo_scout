"""Fallback chains and the circuit breaker, with fake models raising real Gemini error types."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai.errors import ClientError, ServerError
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk, AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from app.config import Settings
from app.llm import (
    ModelProfile,
    ModelSetupError,
    ModelUnavailableError,
    _merge_system_prompt,
    circuit_breaker,
    describe_gemini_error,
    get_chat_model,
    get_guide_model,
    get_structured_model,
    model_chain_problem,
    model_profile,
    next_quota_reset,
    validate_model_chains,
)
from app.main import app
from tests.conftest import FakeGemini

pytestmark = pytest.mark.usefixtures("fresh_settings")

DAILY_QUOTA_DETAILS = [
    {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
    }
]
JUNE_NOON_UTC = datetime(2026, 6, 15, 12, tzinfo=timezone.utc).timestamp()
JUNE_16_MIDNIGHT_PACIFIC = datetime(2026, 6, 16, 7, tzinfo=timezone.utc).timestamp()


def gemini_error(status: int, details: list[dict[str, Any]] | None = None) -> Exception:
    """Build the error the Google client raises for a status."""
    kind = ClientError if status < 500 else ServerError
    return kind(status, {"error": {"code": status, "message": "test", "status": "TEST", "details": details or []}})


class Answer(BaseModel):
    """A schema for the structured output tests."""

    text: str


class Models:
    """A fake provider whose models fail as a test says, recording which ones were called."""

    def __init__(self) -> None:
        self.errors: dict[str, Exception] = {}
        self.calls: list[str] = []
        self.prompts: dict[str, list[BaseMessage]] = {}

    def build(self, name: str) -> "ScriptedModel":
        return ScriptedModel(responses=[f"answer from {name}"], label=name, script=self)


class ScriptedModel(FakeListChatModel):
    """Answers with its own name, unless the test scripted an error for it."""

    label: str
    script: Any

    def _begin(self, messages: list[BaseMessage]) -> None:
        self.script.calls.append(self.label)
        self.script.prompts[self.label] = messages
        if self.label in self.script.errors:
            raise self.script.errors[self.label]

    def _call(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> str:
        self._begin(messages)
        return super()._call(messages, *args, **kwargs)

    async def _astream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        self._begin(messages)
        async for chunk in super()._astream(messages, *args, **kwargs):
            yield chunk

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        return RunnableLambda(lambda model_input: schema(text=self.invoke(model_input).text))


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> Models:
    """Use scripted models in chains of a, b and c."""
    scripted = Models()
    monkeypatch.setattr("app.llm._fake_chat_model", scripted.build)
    monkeypatch.setenv("CHAT_MODELS", "a,b,c")
    monkeypatch.setenv("GUIDE_MODELS", "c,b")
    return scripted


class Clock:
    """A clock that tests move by hand."""

    def __init__(self) -> None:
        self.now = JUNE_NOON_UTC

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Replace the clock the circuit breaker reads."""
    fake_clock = Clock()
    monkeypatch.setattr("app.llm._now", fake_clock)
    return fake_clock


def run_invoke(chain: Runnable) -> str:
    return chain.invoke("Hi").text


def run_ainvoke(chain: Runnable) -> str:
    return asyncio.run(chain.ainvoke("Hi")).text


def run_astream(chain: Runnable) -> str:
    async def collect() -> str:
        return "".join([chunk.text async for chunk in chain.astream("Hi")])

    return asyncio.run(collect())


RUNNERS = [run_invoke, run_ainvoke, run_astream]


# --- settings


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a,b,c", ["a", "b", "c"]),
        (" a , b ,, ", ["a", "b"]),
        ("a,b,a", ["a", "b"]),
        ("", []),
    ],
)
def test_model_lists_come_from_comma_separated_settings(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: list[str]
) -> None:
    monkeypatch.setenv("CHAT_MODELS", value)
    monkeypatch.setenv("GUIDE_MODELS", value)

    settings = Settings(_env_file=None)

    assert settings.chat_models == expected
    assert settings.guide_models == expected


def test_fake_provider_needs_no_model_names() -> None:
    assert get_chat_model().invoke("Hi").text
    assert get_guide_model().invoke("Hi").text


# --- what falls back


@pytest.mark.parametrize("run", RUNNERS)
@pytest.mark.parametrize("status", [404, 429, 500, 503, 504])
def test_unavailable_models_fall_back_to_the_next_model(
    models: Models, run: Callable[[Runnable], str], status: int
) -> None:
    models.errors["a"] = gemini_error(status)

    assert run(get_chat_model()) == "answer from b"
    assert models.calls == ["a", "b"]


@pytest.mark.parametrize("run", RUNNERS)
def test_several_models_can_fail_in_a_row(models: Models, run: Callable[[Runnable], str]) -> None:
    models.errors = {"a": gemini_error(429), "b": gemini_error(503)}

    assert run(get_chat_model()) == "answer from c"
    assert models.calls == ["a", "b", "c"]


@pytest.mark.parametrize("run", RUNNERS)
@pytest.mark.parametrize("status", [400, 401, 403, 502])
def test_other_errors_never_fall_back(models: Models, run: Callable[[Runnable], str], status: int) -> None:
    models.errors["a"] = gemini_error(status)

    with pytest.raises((ClientError, ServerError)) as error:
        run(get_chat_model())

    assert error.value.code == status
    assert models.calls == ["a"]


@pytest.mark.parametrize("run", RUNNERS)
def test_network_errors_do_not_fall_back(models: Models, run: Callable[[Runnable], str]) -> None:
    models.errors["a"] = httpx.ConnectError("no route")

    with pytest.raises(httpx.ConnectError):
        run(get_chat_model())

    assert models.calls == ["a"]


def test_a_400_later_in_the_chain_stops_it(models: Models) -> None:
    models.errors = {"a": gemini_error(503), "b": gemini_error(400)}

    with pytest.raises(ClientError):
        get_chat_model().invoke("Hi")

    assert models.calls == ["a", "b"]


@pytest.mark.parametrize("run", RUNNERS)
def test_when_every_model_is_unavailable_the_message_says_so(models: Models, run: Callable[[Runnable], str]) -> None:
    models.errors = {name: gemini_error(503) for name in "abc"}

    with pytest.raises(ModelUnavailableError) as error:
        run(get_chat_model())

    assert models.calls == ["a", "b", "c"]
    assert "Every configured model" in describe_gemini_error(error.value)


def test_guides_use_their_own_chain(models: Models) -> None:
    models.errors["c"] = gemini_error(503)

    assert get_guide_model().invoke("Hi").text == "answer from b"
    assert models.calls == ["c", "b"]


# --- which model answered


@pytest.mark.parametrize("run", RUNNERS)
def test_the_answering_model_is_reported_and_logged(
    models: Models, run: Callable[[Runnable], str], caplog: pytest.LogCaptureFixture
) -> None:
    models.errors["a"] = gemini_error(503)
    answered_by: list[str] = []

    with caplog.at_level(logging.INFO, logger="app.llm"):
        run(get_chat_model(on_answer=answered_by.append))

    assert answered_by == ["b"]
    assert "Model b answered." in caplog.text
    assert "Model a is unavailable (HTTP 503)" in caplog.text


# --- circuit breaker


def test_daily_quota_skips_the_model_until_midnight_pacific(models: Models, clock: Clock) -> None:
    models.errors["a"] = gemini_error(429, DAILY_QUOTA_DETAILS)
    get_chat_model().invoke("Hi")
    models.calls.clear()

    clock.now = JUNE_16_MIDNIGHT_PACIFIC - 1
    assert get_chat_model().invoke("Hi").text == "answer from b"
    assert models.calls == ["b"]

    models.calls.clear()
    models.errors.clear()
    clock.now = JUNE_16_MIDNIGHT_PACIFIC
    assert get_chat_model().invoke("Hi").text == "answer from a"
    assert models.calls == ["a"]


def test_rate_limit_skips_the_model_for_about_a_minute(models: Models, clock: Clock) -> None:
    models.errors["a"] = gemini_error(429)
    get_chat_model().invoke("Hi")
    models.calls.clear()

    clock.now += 59
    assert get_chat_model().invoke("Hi").text == "answer from b"
    assert models.calls == ["b"]

    models.calls.clear()
    models.errors.clear()
    clock.now += 2
    assert get_chat_model().invoke("Hi").text == "answer from a"


def test_rate_limit_uses_the_delay_gemini_suggests(models: Models, clock: Clock) -> None:
    retry_info = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "20s"}
    models.errors["a"] = gemini_error(429, [retry_info])
    get_chat_model().invoke("Hi")

    clock.now += 19
    assert circuit_breaker.is_open("a")
    clock.now += 2
    assert not circuit_breaker.is_open("a")


@pytest.mark.parametrize("status", [400, 500, 503, 504])
def test_other_errors_do_not_open_the_breaker(models: Models, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setenv("CHAT_MODELS", "a")
    models.errors["a"] = gemini_error(status)

    with pytest.raises((ModelUnavailableError, ClientError)):
        get_chat_model().invoke("Hi")

    assert not circuit_breaker.is_open("a")


@pytest.mark.parametrize("run", RUNNERS)
def test_a_404_disables_the_model_until_restart_and_logs_an_error(
    models: Models, clock: Clock, caplog: pytest.LogCaptureFixture, run: Callable[[Runnable], str]
) -> None:
    models.errors["a"] = gemini_error(404)

    assert run(get_chat_model()) == "answer from b"

    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert errors == ["Model a was not found or is unavailable (HTTP 404). Disabled until restart."]
    models.calls.clear()
    models.errors.clear()
    clock.now += 10 * 365 * 24 * 3600
    assert run(get_chat_model()) == "answer from b"
    assert models.calls == ["b"]


def test_a_restart_forgets_a_disabled_model(models: Models) -> None:
    models.errors["a"] = gemini_error(404)
    get_chat_model().invoke("Hi")
    models.errors.clear()

    circuit_breaker.reset()

    assert get_chat_model().invoke("Hi").text == "answer from a"


def test_a_chain_with_every_model_disabled_fails_health(models: Models, fake_gemini: FakeGemini) -> None:
    for name in ["chat-model-name", "guide-model-name"]:
        circuit_breaker.disable(name)

    problem = model_chain_problem()

    assert "CHAT_MODELS has no usable models" in problem
    assert "GUIDE_MODELS has no usable models" in problem
    with pytest.raises(ModelSetupError, match="restart the server"):
        get_chat_model()


def test_a_disabled_model_leaves_the_last_working_model_retrying(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAT_MODELS", "first,gone")
    circuit_breaker.disable("gone")
    fake_gemini.statuses_by_model["first"] = 503

    with pytest.raises(ModelUnavailableError):
        get_chat_model().invoke("Hi")

    assert fake_gemini.models_called() == ["first", "first"]  # MAX_RETRIES is 1


def test_the_breaker_is_shared_between_chains(models: Models, clock: Clock) -> None:
    models.errors["c"] = gemini_error(429, DAILY_QUOTA_DETAILS)
    get_guide_model().invoke("Hi")
    models.calls.clear()
    models.errors.clear()
    models.errors["a"] = gemini_error(503)
    models.errors["b"] = gemini_error(503)

    with pytest.raises(ModelUnavailableError):
        get_chat_model().invoke("Hi")

    assert models.calls == ["a", "b"]  # c is out of quota, so it gets no call


def test_a_chain_with_every_model_skipped_makes_no_calls(models: Models, clock: Clock) -> None:
    for name in "abc":
        circuit_breaker.open_until(name, clock.now + 60)

    with pytest.raises(ModelUnavailableError) as error:
        get_chat_model().invoke("Hi")

    assert models.calls == []
    assert "Every configured model" in describe_gemini_error(error.value)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        # Summer: Pacific is UTC-7.
        (datetime(2026, 6, 15, 12, tzinfo=timezone.utc), datetime(2026, 6, 16, 7, tzinfo=timezone.utc)),
        # Winter: Pacific is UTC-8.
        (datetime(2026, 1, 15, 12, tzinfo=timezone.utc), datetime(2026, 1, 16, 8, tzinfo=timezone.utc)),
        # A moment before midnight in Pacific time, on the next day in UTC.
        (datetime(2026, 6, 16, 6, 59, tzinfo=timezone.utc), datetime(2026, 6, 16, 7, tzinfo=timezone.utc)),
        # Exactly midnight means the next one.
        (datetime(2026, 6, 16, 7, tzinfo=timezone.utc), datetime(2026, 6, 17, 7, tzinfo=timezone.utc)),
        # The day the clocks change: that day began at UTC-8, and the next begins at UTC-7.
        (datetime(2026, 3, 8, 12, tzinfo=timezone.utc), datetime(2026, 3, 9, 7, tzinfo=timezone.utc)),
    ],
)
def test_next_quota_reset(now: datetime, expected: datetime) -> None:
    assert next_quota_reset(now.timestamp()) == expected.timestamp()


# --- streaming


class StreamingModel(FakeListChatModel):
    """Yields scripted pieces. An exception in the script is raised at that point."""

    pieces: list[Any]

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        for piece in self.pieces:
            if isinstance(piece, Exception):
                raise piece
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


async def collect_stream(chain: Runnable) -> tuple[list[str], Exception | None]:
    """Return the text pieces received, and the error that ended the stream, if any."""
    pieces: list[str] = []
    try:
        async for chunk in chain.astream("Hi"):
            pieces.append(chunk.text)
    except Exception as error:
        return pieces, error
    return pieces, None


def use_streaming_models(monkeypatch: pytest.MonkeyPatch, scripts: dict[str, list[Any]]) -> list[str]:
    """Use a chain of the given models, and return a list that records which ones are built."""
    built: list[str] = []

    def build(name: str) -> StreamingModel:
        built.append(name)
        return StreamingModel(responses=["unused"], pieces=scripts[name])

    monkeypatch.setattr("app.llm._fake_chat_model", build)
    monkeypatch.setenv("CHAT_MODELS", ",".join(scripts))
    return built


def test_an_error_after_the_first_token_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    error = gemini_error(503)
    built = use_streaming_models(monkeypatch, {"a": ["Hello", error], "b": ["from b"]})

    pieces, raised = asyncio.run(collect_stream(get_chat_model()))

    assert pieces == ["Hello"]
    assert raised is error
    assert built == ["a"]


def test_an_error_before_the_first_token_falls_back_even_after_empty_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    built = use_streaming_models(monkeypatch, {"a": ["", "", gemini_error(503)], "b": ["from b"]})
    answered_by: list[str] = []

    pieces, raised = asyncio.run(collect_stream(get_chat_model(on_answer=answered_by.append)))

    assert "".join(pieces) == "from b"
    assert raised is None
    assert built == ["a", "b"]
    assert answered_by == ["b"]


def test_a_rate_limit_after_the_first_token_still_opens_the_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    use_streaming_models(monkeypatch, {"a": ["Hello", gemini_error(429)], "b": ["from b"]})

    asyncio.run(collect_stream(get_chat_model()))

    assert circuit_breaker.is_open("a")


def test_a_model_that_returns_nothing_still_counts_as_answering(monkeypatch: pytest.MonkeyPatch) -> None:
    built = use_streaming_models(monkeypatch, {"a": [""], "b": ["from b"]})
    answered_by: list[str] = []

    pieces, raised = asyncio.run(collect_stream(get_chat_model(on_answer=answered_by.append)))

    assert "".join(pieces) == ""
    assert raised is None
    assert built == ["a"]
    assert answered_by == ["a"]


def test_models_are_only_built_when_their_turn_comes(monkeypatch: pytest.MonkeyPatch) -> None:
    built = use_streaming_models(monkeypatch, {"a": ["from a"], "b": ["from b"]})

    asyncio.run(collect_stream(get_chat_model()))

    assert built == ["a"]


# --- per-model settings


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gemini-3.5-flash-lite", ModelProfile(thinking="level")),
        ("gemini-3.1-flash-lite", ModelProfile(thinking="level")),
        ("gemini-3-flash", ModelProfile(thinking="level")),
        ("gemini-3.8-flash", ModelProfile(thinking="level")),
        ("models/gemini-3-flash", ModelProfile(thinking="level")),
        ("gemini-2.5-flash-lite", ModelProfile(thinking="budget")),
        ("gemini-2.0-flash", ModelProfile()),
        ("gemma-4-31b-it", ModelProfile(system_prompt=False, structured_output=False)),
        ("some-new-model", ModelProfile(structured_output=False)),
        ("fake-model", ModelProfile()),
    ],
)
def test_model_profiles(name: str, expected: ModelProfile) -> None:
    assert model_profile(name) == expected


def test_merge_puts_the_system_prompt_in_the_first_user_message() -> None:
    messages = [
        SystemMessage("Rules."),
        HumanMessage("Old question"),
        AIMessage("Old answer"),
        HumanMessage("New question"),
    ]

    merged = _merge_system_prompt(messages)

    assert [(type(message), message.text) for message in merged] == [
        (HumanMessage, "Rules.\n\nOld question"),
        (AIMessage, "Old answer"),
        (HumanMessage, "New question"),
    ]


def test_merge_leaves_input_without_a_system_prompt_alone() -> None:
    assert _merge_system_prompt("Hi") == "Hi"
    assert _merge_system_prompt([HumanMessage("Hi")]) == [HumanMessage("Hi")]


def test_merge_without_a_user_message_makes_one() -> None:
    assert _merge_system_prompt([SystemMessage("Rules.")]) == [HumanMessage("Rules.")]


def test_only_models_without_system_prompts_get_a_merged_prompt(
    models: Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAT_MODELS", "gemini-3-flash,gemma-4-31b-it")
    models.errors["gemini-3-flash"] = gemini_error(503)
    prompt = [SystemMessage("Rules."), HumanMessage("Question")]

    get_chat_model().invoke(prompt)

    assert isinstance(models.prompts["gemini-3-flash"][0], SystemMessage)
    assert [message.text for message in models.prompts["gemma-4-31b-it"]] == ["Rules.\n\nQuestion"]


# --- what really goes over the wire, using the real Gemini client against a local server


def request_for(fake_gemini: FakeGemini, model: str) -> dict[str, Any]:
    """Return the body of the first request to a model."""
    return next(body for name, body in fake_gemini.requests if name == model)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3-flash", {"thinking_level": "MINIMAL"}),
        ("gemini-2.5-flash-lite", {"thinking_budget": 0}),
        ("gemma-4-31b-it", None),
        ("antigravity-preview-09-2026", None),
    ],
)
def test_each_family_gets_its_own_thinking_option(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, model: str, expected: dict[str, Any] | None
) -> None:
    monkeypatch.setenv("CHAT_MODELS", model)
    monkeypatch.setenv("CHAT_THINKING_LEVEL", "minimal")

    get_chat_model().invoke("Hi")

    config = request_for(fake_gemini, model).get("generationConfig", {}).get("thinkingConfig")
    assert config == expected


def test_chat_and_guide_chains_use_their_own_thinking_levels(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAT_MODELS", "gemini-3-chat")
    monkeypatch.setenv("GUIDE_MODELS", "gemini-3-guide")
    monkeypatch.setenv("CHAT_THINKING_LEVEL", "minimal")
    monkeypatch.setenv("GUIDE_THINKING_LEVEL", "high")

    get_chat_model().invoke("Hi")
    get_guide_model().invoke("Hi")

    assert request_for(fake_gemini, "gemini-3-chat")["generationConfig"]["thinkingConfig"] == {
        "thinking_level": "MINIMAL"
    }
    assert request_for(fake_gemini, "gemini-3-guide")["generationConfig"]["thinkingConfig"] == {
        "thinking_level": "HIGH"
    }


@pytest.mark.parametrize("model", ["gemini-3-flash", "gemini-2.5-flash-lite"])
def test_default_thinking_sends_nothing_on_any_family(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("CHAT_MODELS", model)
    monkeypatch.setenv("CHAT_THINKING_LEVEL", "default")

    get_chat_model().invoke("Hi")

    assert "thinkingConfig" not in request_for(fake_gemini, model).get("generationConfig", {})


def test_gemma_requests_have_no_system_instruction(fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_MODELS", "gemini-3-flash,gemma-4-31b-it")
    fake_gemini.statuses_by_model["gemini-3-flash"] = 503
    prompt = [SystemMessage("Rules."), HumanMessage("Question")]

    get_chat_model().invoke(prompt)

    gemini = request_for(fake_gemini, "gemini-3-flash")
    assert gemini["systemInstruction"]["parts"][0]["text"] == "Rules."
    gemma = request_for(fake_gemini, "gemma-4-31b-it")
    assert "systemInstruction" not in gemma
    assert gemma["contents"][0]["parts"][0]["text"] == "Rules.\n\nQuestion"


def test_real_client_errors_fall_back_and_only_the_last_model_retries(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAT_MODELS", "first,second,last")
    fake_gemini.statuses_by_model = {"first": 429, "second": 503, "last": 503}

    with pytest.raises(ModelUnavailableError):
        get_chat_model().invoke("Hi")

    assert fake_gemini.models_called() == ["first", "second", "last", "last"]  # MAX_RETRIES is 1
    assert circuit_breaker.is_open("first")
    assert not circuit_breaker.is_open("second")


@pytest.mark.parametrize("run", RUNNERS)
def test_real_client_daily_quota_opens_the_breaker_until_the_reset(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, clock: Clock, run: Callable[[Runnable], str]
) -> None:
    monkeypatch.setenv("CHAT_MODELS", "first,second")
    fake_gemini.statuses_by_model["first"] = 429
    fake_gemini.error_details = DAILY_QUOTA_DETAILS

    assert run(get_chat_model()) == "hello"
    assert run(get_chat_model()) == "hello"

    assert fake_gemini.models_called() == ["first", "second", "second"]
    clock.now = JUNE_16_MIDNIGHT_PACIFIC - 1
    assert circuit_breaker.is_open("first")
    clock.now = JUNE_16_MIDNIGHT_PACIFIC
    assert not circuit_breaker.is_open("first")


@pytest.mark.parametrize("status", [404, 500, 503, 504])
@pytest.mark.parametrize("run", RUNNERS)
def test_real_client_errors_fall_back_on_every_path(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, run: Callable[[Runnable], str], status: int
) -> None:
    monkeypatch.setenv("CHAT_MODELS", "first,second")
    monkeypatch.setenv("MAX_RETRIES", "0")
    fake_gemini.statuses_by_model["first"] = status

    assert run(get_chat_model()) == "hello"
    assert fake_gemini.models_called() == ["first", "second"]


@pytest.mark.parametrize("run", RUNNERS)
def test_real_client_400_does_not_fall_back(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, run: Callable[[Runnable], str]
) -> None:
    monkeypatch.setenv("CHAT_MODELS", "first,second")
    fake_gemini.statuses_by_model["first"] = 400

    with pytest.raises(Exception) as error:
        run(get_chat_model())

    assert not isinstance(error.value, ModelUnavailableError)
    assert "rejected the request" in describe_gemini_error(error.value)
    assert fake_gemini.models_called() == ["first"]


# --- checking model names at startup


def list_models(fake_gemini: FakeGemini, *names: str) -> None:
    """Make the model list contain these names, all able to generate text."""
    fake_gemini.listed_models = {name: ["generateContent"] for name in names}


@pytest.fixture
def chains(fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch) -> FakeGemini:
    """Configure a chat chain of wrong-name and real-name, and a guide chain of real-name."""
    monkeypatch.setenv("CHAT_MODELS", "wrong-name,real-name")
    monkeypatch.setenv("GUIDE_MODELS", "real-name")
    return fake_gemini


def test_unknown_names_are_dropped_with_a_warning(chains: FakeGemini, caplog: pytest.LogCaptureFixture) -> None:
    list_models(chains, "real-name")

    validate_model_chains()

    get_chat_model().invoke("Hi")
    assert chains.models_called() == ["real-name"]
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "CHAT_MODELS" in warnings[0] and "'wrong-name'" in warnings[0]
    assert model_chain_problem() is None


def test_model_names_can_have_the_models_prefix(chains: FakeGemini, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_MODELS", "models/real-name")
    list_models(chains, "real-name")

    validate_model_chains()

    assert model_chain_problem() is None


def test_a_model_that_cannot_generate_text_counts_as_unknown(chains: FakeGemini) -> None:
    chains.listed_models = {"real-name": ["generateContent"], "wrong-name": ["embedContent"]}

    validate_model_chains()

    get_chat_model().invoke("Hi")
    assert chains.models_called() == ["real-name"]


def test_validating_again_forgets_old_results(chains: FakeGemini) -> None:
    list_models(chains, "real-name")
    validate_model_chains()
    list_models(chains, "real-name", "wrong-name")

    validate_model_chains()

    get_chat_model().invoke("Hi")
    assert chains.models_called() == ["wrong-name"]


def test_startup_finding_empty_chains_fails_health_and_names_them(chains: FakeGemini) -> None:
    list_models(chains, "something-else")

    with TestClient(app) as started:
        response = started.get("/api/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["db"] == "ok"
    assert "CHAT_MODELS has no usable models" in body["models"]
    assert "GUIDE_MODELS has no usable models" in body["models"]
    assert "wrong-name" in body["models"]


def test_startup_with_usable_chains_passes_health(chains: FakeGemini) -> None:
    list_models(chains, "real-name")

    with TestClient(app) as started:
        response = started.get("/api/health")

    assert response.status_code == 200
    assert response.json()["models"] == "ok"


def test_an_empty_chain_explains_itself_when_used(chains: FakeGemini) -> None:
    list_models(chains, "something-else")
    validate_model_chains()

    with pytest.raises(ModelSetupError, match="CHAT_MODELS has no usable models"):
        get_chat_model()
    with pytest.raises(ModelSetupError, match="GUIDE_MODELS has no usable models"):
        get_guide_model()


def test_one_empty_chain_fails_health_and_names_only_that_one(
    chains: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GUIDE_MODELS", "wrong-name")
    list_models(chains, "real-name")

    validate_model_chains()

    problem = model_chain_problem()
    assert "GUIDE_MODELS has no usable models" in problem
    assert "CHAT_MODELS" not in problem


@pytest.mark.parametrize("status", [400, 503])
def test_a_failed_model_list_keeps_every_model(
    chains: FakeGemini, caplog: pytest.LogCaptureFixture, status: int
) -> None:
    chains.list_status = status

    validate_model_chains()

    assert model_chain_problem() is None
    assert "Couldn't check the model names" in caplog.text
    get_chat_model().invoke("Hi")
    assert chains.models_called() == ["wrong-name"]


def test_listing_models_makes_no_generation_calls(chains: FakeGemini) -> None:
    list_models(chains, "real-name")

    validate_model_chains()

    assert chains.requests == []


def test_the_fake_provider_skips_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(api_key: str) -> set[str]:
        raise AssertionError("the fake provider must stay offline")

    monkeypatch.setattr("app.llm._list_generation_models", fail)
    monkeypatch.setenv("CHAT_MODELS", "anything")

    validate_model_chains()

    assert model_chain_problem() is None


@pytest.mark.parametrize("variable", ["GOOGLE_API_KEY", "CHAT_MODELS", "GUIDE_MODELS"])
def test_missing_settings_fail_health(fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.setenv(variable, "")

    assert f"{variable} is not set" in model_chain_problem()


# --- structured output


def test_structured_output_skips_models_that_cannot_do_it(models: Models, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_MODELS", "gemma-4-31b-it,gemini-3-flash,antigravity-preview-09-2026,gemini-2.5-flash-lite")
    models.errors["gemini-3-flash"] = gemini_error(503)

    assert get_structured_model(Answer).invoke("Hi") == Answer(text="answer from gemini-2.5-flash-lite")
    assert models.calls == ["gemini-3-flash", "gemini-2.5-flash-lite"]


def test_structured_output_does_not_fall_back_on_a_400(models: Models, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_MODELS", "gemini-3-flash,gemini-2.5-flash-lite")
    models.errors["gemini-3-flash"] = gemini_error(400)

    with pytest.raises(ClientError):
        get_structured_model(Answer).invoke("Hi")

    assert models.calls == ["gemini-3-flash"]


def test_structured_output_with_no_capable_model_explains_itself(
    models: Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAT_MODELS", "gemma-4-31b-it,antigravity-preview-09-2026")

    with pytest.raises(ValueError, match="supports structured output"):
        get_structured_model(Answer)


def test_real_client_structured_output_asks_for_the_schema(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAT_MODELS", "gemini-3-flash")

    with pytest.raises(Exception):  # the fake server's reply isn't valid JSON for the schema
        get_structured_model(Answer).invoke("Hi")

    config = request_for(fake_gemini, "gemini-3-flash")["generationConfig"]
    assert config["responseMimeType"] == "application/json"
