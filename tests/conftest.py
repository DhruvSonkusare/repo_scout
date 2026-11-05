"""Shared test setup: every test runs against the real repoguide_test Postgres database."""

import json
import os
import shutil
import ssl
import tempfile
import threading
import time
from collections.abc import Iterator
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# This must run before anything imports the app, because app.db builds its engine on import.
load_dotenv(PROJECT_ROOT / ".env")
test_database_url = os.environ.get("TEST_DATABASE_URL")
if not test_database_url:
    raise RuntimeError("Set TEST_DATABASE_URL in .env or the environment to run the tests.")
# Tests truncate every table, so refuse anything that isn't clearly a test database.
if not (make_url(test_database_url).database or "").endswith("_test"):
    raise RuntimeError("TEST_DATABASE_URL must point at a database whose name ends in _test.")
os.environ["DATABASE_URL"] = test_database_url
# Tests never call a real model, so they need no API key or quota.
os.environ["LLM_PROVIDER"] = "fake"
os.environ["EMBEDDING_PROVIDER"] = "fake"
# Clones and vectors go in a temporary folder, so tests never touch real data.
TEST_ROOT = Path(tempfile.mkdtemp(prefix="repoguide-tests-"))
os.environ["DATA_DIR"] = str(TEST_ROOT / "data")

from app.config import get_settings  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.ingest import remove_clone  # noqa: E402

SAMPLE_REPO = PROJECT_ROOT / "tests" / "fixtures" / "sample_repo"


@pytest.fixture(scope="session", autouse=True)
def remove_test_root() -> Iterator[None]:
    """Delete the temporary data folder after the test session."""
    yield
    # Chroma can keep its files open until the process exits, so this can't always finish.
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def alembic_config() -> Config:
    """Alembic configuration for this project."""
    return Config(str(PROJECT_ROOT / "alembic.ini"))


@pytest.fixture(scope="session", autouse=True)
def migrated_database(alembic_config: Config) -> None:
    """Build the schema with the real migrations once per test session."""
    command.upgrade(alembic_config, "head")


@pytest.fixture(autouse=True)
def clean_tables(migrated_database: None) -> None:
    """Start each test with empty tables and ids counting from 1."""
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    """Let a test change settings through the environment without affecting other tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def db_session() -> Iterator[Session]:
    """A session on the test database."""
    with SessionLocal() as session:
        yield session


@pytest.fixture
def cloned_urls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace git clone with a copy of the sample repo, and record the URLs it was asked for."""
    urls: list[str] = []

    def copy_sample(url: str, repo_id: int) -> Path:
        urls.append(url)
        remove_clone(repo_id)
        destination = get_settings().repos_dir / str(repo_id)
        shutil.copytree(SAMPLE_REPO, destination)
        return destination

    monkeypatch.setattr("app.ingest.clone_repo", copy_sample)
    return urls


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace time.sleep so tests don't really wait, and record the requested delays.

    The Google SDK's backoff also sleeps through time.sleep, so this covers its retries too.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


_cached_ssl_context = lru_cache(maxsize=None)(ssl.create_default_context)


class FakeGemini:
    """A local stand-in for the Gemini API, answering with a scripted list of statuses."""

    def __init__(self, sleeps: list[float]) -> None:
        self.statuses = [200]  # one per request; the last one repeats
        self.error_details: list[dict[str, Any]] = []
        self.attempts = 0
        self.batch_sizes: list[int] = []  # texts per successful embedding request
        self.sleeps = sleeps

    def next_status(self) -> int:
        """Count this attempt and return the status to answer it with."""
        self.attempts += 1
        return self.statuses[min(self.attempts, len(self.statuses)) - 1]

    def response_body(self, status: int, request: dict[str, Any]) -> dict[str, Any]:
        """Build a chat and embedding success response, or an error response."""
        if status != 200:
            return {"error": {"code": status, "message": "test error", "status": "TEST", "details": self.error_details}}
        text_count = len(request.get("requests", [])) or 1
        self.batch_sizes.append(text_count)
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
            "embedding": {"values": [0.1, 0.2]},
            "embeddings": [{"values": [0.1, 0.2]}] * text_count,
        }


@pytest.fixture
def fake_gemini(
    monkeypatch: pytest.MonkeyPatch, recorded_sleeps: list[float], fresh_settings: None
) -> Iterator[FakeGemini]:
    """Run the local server and point the Gemini provider at it, with one retry and no real sleeping."""
    gemini = FakeGemini(recorded_sleeps)
    # Each Google client loads the TLS certificates (about 0.5s), so reuse them. The server is plain HTTP.
    monkeypatch.setattr(ssl, "create_default_context", _cached_ssl_context)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            status = gemini.next_status()
            body = json.dumps(gemini.response_body(status, request)).encode()
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
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder-key")
    monkeypatch.setenv("CHAT_MODEL", "chat-model-name")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model-name")
    monkeypatch.setenv("MAX_RETRIES", "1")
    yield gemini
    server.shutdown()
    server.server_close()
