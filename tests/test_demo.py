"""Tests for the public demo's limits, and for what the deployment files promise."""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import DailyUsage, Repo
from app.demo import client_ip
from app.main import app
from scripts.index_demo_repos import DEPLOYED_EMBEDDING_MODEL, DEPLOYED_EMBEDDING_PROVIDER, MAX_TOTAL_CHUNKS

pytestmark = pytest.mark.usefixtures("fresh_settings", "cloned_urls")

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)
PASSWORD = "the-index-password"
GOOD = {"X-Index-Password": PASSWORD}
URL = "https://github.com/acme/sample"


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Turn demo mode on, with a small repo and no GitHub calls."""
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("INDEX_PASSWORD", PASSWORD)
    monkeypatch.setattr("app.demo._repo_size_mb", lambda name: 1.0)
    return monkeypatch


def add(headers: dict[str, str] | None = None, url: str = URL) -> Any:
    return client.post("/api/repos", json={"url": url}, headers=headers or {})


# --- with demo mode off, nothing changes


def test_no_password_or_limit_applies_outside_demo_mode() -> None:
    assert add().status_code == 201
    for _ in range(3):
        assert client.post("/api/repos/1/ask", json={"question": "Hi?"}).status_code == 200
    assert client.delete("/api/repos/1").status_code == 204


# --- the index password


def test_adding_a_repo_needs_the_password(demo: pytest.MonkeyPatch, db_session: Session) -> None:
    response = add()

    assert response.status_code == 401
    assert "index password" in response.json()["detail"]
    assert db_session.query(Repo).count() == 0


def test_a_wrong_password_is_refused(demo: pytest.MonkeyPatch) -> None:
    assert add({"X-Index-Password": "wrong"}).status_code == 401


def test_the_right_password_adds_the_repo(demo: pytest.MonkeyPatch, cloned_urls: list[str]) -> None:
    assert add(GOOD).status_code == 201
    assert cloned_urls == [URL]


def test_a_repo_already_here_needs_no_password(demo: pytest.MonkeyPatch) -> None:
    add(GOOD)

    assert add().status_code == 200


@pytest.mark.parametrize(("method", "suffix"), [("post", "/reindex"), ("delete", "")])
def test_changing_a_demo_repo_needs_the_password(demo: pytest.MonkeyPatch, method: str, suffix: str) -> None:
    add(GOOD)

    assert client.request(method, f"/api/repos/1{suffix}").status_code == 401
    assert client.get("/api/repos/1").json()["status"] == "ready"
    assert client.request(method, f"/api/repos/1{suffix}", headers=GOOD).status_code in (202, 204)


def test_no_password_configured_turns_indexing_off(demo: pytest.MonkeyPatch) -> None:
    demo.setenv("INDEX_PASSWORD", "")

    response = add({"X-Index-Password": ""})

    assert response.status_code == 403
    assert "turned off" in response.json()["detail"]


def test_password_guesses_are_rate_limited(demo: pytest.MonkeyPatch) -> None:
    demo.setenv("RATE_LIMIT_INDEX_PER_HOUR", "3")

    codes = [add({"X-Index-Password": f"guess-{n}"}).status_code for n in range(4)]

    assert codes == [401, 401, 401, 429]


# --- repo limits


def test_a_repo_over_the_size_limit_is_refused(demo: pytest.MonkeyPatch, cloned_urls: list[str]) -> None:
    demo.setattr("app.demo._repo_size_mb", lambda name: 500.0)

    response = add(GOOD)

    assert response.status_code == 413
    assert "20 MB" in response.json()["detail"]
    assert cloned_urls == []


def test_a_repo_with_too_many_chunks_fails_before_using_the_embedding_quota(demo: pytest.MonkeyPatch) -> None:
    demo.setenv("MAX_INDEX_CHUNKS", "1")

    add(GOOD)

    repo = client.get("/api/repos/1").json()
    assert repo["status"] == "failed"
    assert "daily embedding quota" in repo["error"]
    assert repo["chunks_embedded"] == 0


# --- rate limits and the daily cap


@pytest.mark.parametrize(
    ("setting", "path", "body", "word"),
    [
        ("RATE_LIMIT_ASK_PER_HOUR", "/api/repos/1/ask", {"question": "Hi?"}, "questions"),
        ("RATE_LIMIT_GUIDE_PER_HOUR", "/api/repos/1/guide", None, "guides"),
    ],
)
def test_each_client_is_rate_limited(
    demo: pytest.MonkeyPatch, setting: str, path: str, body: dict | None, word: str
) -> None:
    demo.setenv(setting, "2")
    add(GOOD)

    responses = [client.post(path, json=body) for _ in range(3)]

    assert [response.status_code for response in responses] == [*[200, 200], 429]
    assert word in responses[-1].json()["detail"]
    assert "Retry-After" in responses[-1].headers


def test_search_is_rate_limited(demo: pytest.MonkeyPatch) -> None:
    demo.setenv("RATE_LIMIT_SEARCH_PER_HOUR", "1")
    client.post("/api/finder/search", json={"problem": ""})

    response = client.post("/api/finder/search", json={"problem": "Extract tables"})

    assert response.status_code == 429
    assert "searches" in response.json()["detail"]


def test_another_client_has_its_own_limit(demo: pytest.MonkeyPatch) -> None:
    demo.setenv("RATE_LIMIT_ASK_PER_HOUR", "1")
    demo.setenv("CLIENT_IP_HEADER", "true-client-ip")
    add(GOOD)
    ask = lambda ip: client.post("/api/repos/1/ask", json={"question": "Hi?"}, headers={"True-Client-IP": ip}).status_code  # noqa: E731

    assert [ask("1.1.1.1"), ask("1.1.1.1"), ask("2.2.2.2")] == [200, 429, 200]


def test_the_daily_cap_shows_a_friendly_message(demo: pytest.MonkeyPatch, db_session: Session) -> None:
    demo.setenv("DAILY_REQUEST_CAP", "2")
    add(GOOD)

    responses = [client.post("/api/repos/1/ask", json={"question": "Hi?"}) for _ in range(3)]

    assert [response.status_code for response in responses] == [200, 200, 429]
    assert responses[-1].json()["detail"].startswith("Demo limit reached")
    assert db_session.query(DailyUsage).one().requests == 3


def test_the_cap_is_shared_between_clients_and_endpoints(demo: pytest.MonkeyPatch) -> None:
    demo.setenv("DAILY_REQUEST_CAP", "1")
    demo.setenv("CLIENT_IP_HEADER", "true-client-ip")
    add(GOOD)

    first = client.post("/api/repos/1/ask", json={"question": "Hi?"}, headers={"True-Client-IP": "1.1.1.1"})
    second = client.post("/api/repos/1/guide", headers={"True-Client-IP": "2.2.2.2"})

    assert (first.status_code, second.status_code) == (200, 429)


def test_reading_saved_work_costs_nothing(demo: pytest.MonkeyPatch) -> None:
    demo.setenv("DAILY_REQUEST_CAP", "1")
    add(GOOD)

    for _ in range(5):
        assert client.get("/api/repos").status_code == 200
        assert client.get("/api/finder/searches").status_code == 200

    assert client.post("/api/repos/1/ask", json={"question": "Hi?"}).status_code == 200


# --- finding the client's IP


class FakeRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        self.client = type("Client", (), {"host": "10.0.0.1"})()


def test_a_configured_header_gives_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIENT_IP_HEADER", "true-client-ip")

    assert client_ip(FakeRequest({"true-client-ip": "9.9.9.9, 1.1.1.1"})) == "9.9.9.9"


def test_without_a_configured_header_a_client_cannot_choose_its_ip() -> None:
    assert client_ip(FakeRequest({"true-client-ip": "9.9.9.9", "x-forwarded-for": "8.8.8.8"})) == "10.0.0.1"


def test_a_missing_header_falls_back_to_the_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIENT_IP_HEADER", "true-client-ip")

    assert client_ip(FakeRequest({})) == "10.0.0.1"


# --- restoring clones


def test_a_lost_clone_is_restored_for_the_file_viewer(demo: pytest.MonkeyPatch, cloned_urls: list[str]) -> None:
    add(GOOD)

    shutil.rmtree(get_settings().repos_dir / "1")

    response = client.get("/api/repos/1/file", params={"path": "src/app.py"})

    assert response.status_code == 200
    assert "def greet" in response.json()["content"]
    assert cloned_urls == [URL, URL]


def test_a_lost_clone_is_restored_for_guides(demo: pytest.MonkeyPatch) -> None:
    add(GOOD)

    shutil.rmtree(get_settings().repos_dir / "1")

    assert client.post("/api/repos/1/guide").status_code == 200


# --- the health check doesn't reveal connection details in public


def test_public_health_hides_database_errors(demo: pytest.MonkeyPatch) -> None:
    demo.setattr("app.main.engine", create_engine("postgresql+psycopg://secret-user:pw@127.0.0.1:1/secret-db", connect_args={"connect_timeout": 1}))

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json()["db"] == "unavailable"
    assert "secret" not in response.text


def test_config_reports_demo_mode(demo: pytest.MonkeyPatch) -> None:
    assert client.get("/api/config").json() == {"demo_mode": True}


# --- what the deployment files promise


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_the_image_listens_on_the_port_the_host_chooses() -> None:
    dockerfile = read("Dockerfile")

    assert "--port ${PORT:-10000}" in dockerfile
    assert "alembic upgrade head && exec uvicorn" in dockerfile
    assert "7860" not in dockerfile


def test_the_image_runs_as_a_non_root_user_with_git() -> None:
    dockerfile = read("Dockerfile")

    assert re.search(r"^USER appuser$", dockerfile, re.MULTILINE)
    assert "git ca-certificates" in dockerfile


def test_the_image_leaves_out_local_embeddings_to_save_memory() -> None:
    assert "requirements-base.txt" in read("Dockerfile")
    assert "requirements.txt" not in read("Dockerfile")
    packages = [line.lower() for line in read("requirements-base.txt").splitlines() if not line.startswith("#")]
    assert not [line for line in packages if line.startswith(("fastembed", "onnxruntime"))]


def test_importing_the_app_does_not_load_local_embeddings() -> None:
    code = "import sys, app.main; print([name for name in ('fastembed', 'onnxruntime') if name in sys.modules])"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8", cwd=ROOT, check=True)

    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_the_demo_script_uses_the_same_model_as_the_image() -> None:
    dockerfile = read("Dockerfile")

    assert f"EMBEDDING_PROVIDER={DEPLOYED_EMBEDDING_PROVIDER}" in dockerfile
    assert f"EMBEDDING_MODEL={DEPLOYED_EMBEDDING_MODEL}" in dockerfile


def test_the_demo_repos_leave_room_in_the_embedding_quota() -> None:
    assert MAX_TOTAL_CHUNKS < 1000


def test_secrets_are_not_baked_into_the_image_or_blueprint() -> None:
    dockerfile = read("Dockerfile")
    for secret in (
        "GOOGLE_API_KEY", "GITHUB_TOKEN", "DATABASE_URL", "INDEX_PASSWORD",
        "GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET", "SESSION_SECRET", "PUBLIC_BASE_URL",
    ):
        assert secret not in dockerfile
        assert f"key: {secret}\n        sync: false" in read("render.yaml")
    assert ".env" in read(".dockerignore")


def test_the_readme_has_no_hugging_face_frontmatter() -> None:
    readme = read("README.md")

    assert not readme.startswith("---")
    assert "app_port" not in readme


def test_the_new_name_is_used_everywhere_visible() -> None:
    for name in ("static/index.html", "README.md", "docs/DEPLOY.md", "app/main.py"):
        text = read(name)
        assert "RepoScout" in text
        assert "RepoGuide" not in text
