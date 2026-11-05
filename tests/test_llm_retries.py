"""Retry and error-message behaviour, against a local server that acts like the Gemini API.

The fake_gemini fixture patches out sleeping, so backoff is checked without any real waiting.
"""

import pytest

from app.llm import describe_gemini_error, gemini_retry_delay, get_chat_model, get_embeddings
from scripts.smoke_llm import main as smoke_main
from tests.conftest import FakeGemini


@pytest.mark.parametrize("status", [429, 503])
def test_chat_retries_temporary_errors_with_a_backoff(fake_gemini: FakeGemini, status: int) -> None:
    fake_gemini.statuses = [status, 200]

    reply = get_chat_model().invoke("Hello?")

    assert reply.text == "hello"
    assert fake_gemini.attempts == 2
    assert len(fake_gemini.sleeps) == 1 and fake_gemini.sleeps[0] > 0


@pytest.mark.parametrize("status", [429, 503])
def test_embeddings_retry_temporary_errors_with_a_backoff(fake_gemini: FakeGemini, status: int) -> None:
    fake_gemini.statuses = [status, 200]

    vector = get_embeddings().embed_query("Hello?")

    assert vector == [0.1, 0.2]
    assert fake_gemini.attempts == 2
    assert len(fake_gemini.sleeps) == 1 and fake_gemini.sleeps[0] > 0


def test_embedding_batches_retry_temporary_errors(fake_gemini: FakeGemini) -> None:
    fake_gemini.statuses = [503, 200]

    vectors = get_embeddings().embed_documents(["Hello?"])

    assert vectors == [[0.1, 0.2]]
    assert fake_gemini.attempts == 2


def test_permanent_errors_are_not_retried(fake_gemini: FakeGemini) -> None:
    fake_gemini.statuses = [404]

    with pytest.raises(Exception):
        get_chat_model().invoke("Hello?")
    assert fake_gemini.attempts == 1

    fake_gemini.attempts = 0
    with pytest.raises(Exception):
        get_embeddings().embed_query("Hello?")
    assert fake_gemini.attempts == 1
    assert fake_gemini.sleeps == []


def test_zero_max_retries_means_a_single_attempt(fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_RETRIES", "0")
    fake_gemini.statuses = [503]

    with pytest.raises(Exception):
        get_chat_model().invoke("Hello?")
    assert fake_gemini.attempts == 1

    fake_gemini.attempts = 0
    with pytest.raises(Exception):
        get_embeddings().embed_query("Hello?")
    assert fake_gemini.attempts == 1
    assert fake_gemini.sleeps == []


def embedding_error(fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, status: int) -> Exception:
    """Return the real exception the embedding client raises for a status."""
    monkeypatch.setenv("MAX_RETRIES", "0")
    fake_gemini.statuses = [status]
    with pytest.raises(Exception) as error:
        get_embeddings().embed_query("Hello?")
    return error.value


def test_retry_delay_comes_from_gemini_when_it_gives_one(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_gemini.error_details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37.5s"}]

    assert gemini_retry_delay(embedding_error(fake_gemini, monkeypatch, 429), 60) == 37.5


@pytest.mark.parametrize(("status", "expected"), [(429, 60), (503, 60), (400, None), (404, None)])
def test_retry_delay_defaults_and_permanent_errors(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, status: int, expected: float | None
) -> None:
    assert gemini_retry_delay(embedding_error(fake_gemini, monkeypatch, status), 60) == expected


def test_daily_quota_message_says_it_is_daily(fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gemini.error_details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
        }
    ]

    assert "daily quota" in describe_gemini_error(embedding_error(fake_gemini, monkeypatch, 429))


def test_smoke_script_prints_results_on_success(fake_gemini: FakeGemini, capsys: pytest.CaptureFixture[str]) -> None:
    smoke_main()

    output = capsys.readouterr().out
    assert "Chat reply: hello" in output
    assert "Embedding vector length: 2" in output


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([503], "Gemini is busy"),  # the chat call fails
        ([429], "rate limit"),
        ([404], "can't find that model"),
        ([200, 503], "Gemini is busy"),  # chat works, then the embedding call fails
    ],
)
def test_smoke_script_gives_a_short_message_on_api_errors(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, statuses: list[int], expected: str
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "0")
    fake_gemini.statuses = statuses

    with pytest.raises(SystemExit) as exit_info:
        smoke_main()

    message = str(exit_info.value)
    assert expected in message
    assert "\n" not in message


@pytest.mark.usefixtures("fresh_settings")
def test_smoke_script_reports_a_missing_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("CHAT_MODEL", "chat-model-name")

    with pytest.raises(SystemExit, match="GOOGLE_API_KEY"):
        smoke_main()
