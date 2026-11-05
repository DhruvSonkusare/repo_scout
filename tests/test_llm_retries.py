"""Retry and error-message behaviour, against a local server that acts like the Gemini API."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.llm import get_chat_model, get_embeddings
from scripts.smoke_llm import main as smoke_main

pytestmark = pytest.mark.usefixtures("fresh_settings")

SUCCESS_BODY = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
    "embedding": {"values": [0.1, 0.2]},
    "embeddings": [{"values": [0.1, 0.2]}],
}


class FakeGemini:
    """Answers each request with the next status in `statuses`, repeating the last one."""

    def __init__(self) -> None:
        self.statuses = [200]
        self.attempts = 0

    def next_status(self) -> int:
        """Count this attempt and return the status to answer it with."""
        self.attempts += 1
        return self.statuses[min(self.attempts, len(self.statuses)) - 1]


@pytest.fixture
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGemini]:
    """Run the local server and point the Gemini provider at it, with one retry."""
    gemini = FakeGemini()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("content-length", 0)))
            status = gemini.next_status()
            error = {"error": {"code": status, "message": "test error", "status": "TEST"}}
            body = json.dumps(SUCCESS_BODY if status == 200 else error).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            """Keep test output quiet."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # A short poll interval makes server.shutdown() return quickly after each test.
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder-key")
    monkeypatch.setenv("CHAT_MODEL", "chat-model-name")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model-name")
    monkeypatch.setenv("MAX_RETRIES", "1")
    yield gemini
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("status", [429, 503])
def test_chat_retries_temporary_errors(fake_gemini: FakeGemini, status: int) -> None:
    fake_gemini.statuses = [status, 200]

    reply = get_chat_model().invoke("Hello?")

    assert reply.text == "hello"
    assert fake_gemini.attempts == 2


@pytest.mark.parametrize("status", [429, 503])
def test_embeddings_retry_temporary_errors(fake_gemini: FakeGemini, status: int) -> None:
    fake_gemini.statuses = [status, 200]

    vector = get_embeddings().embed_query("Hello?")

    assert vector == [0.1, 0.2]
    assert fake_gemini.attempts == 2


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


def test_smoke_script_reports_a_missing_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("CHAT_MODEL", "chat-model-name")

    with pytest.raises(SystemExit, match="GOOGLE_API_KEY"):
        smoke_main()
