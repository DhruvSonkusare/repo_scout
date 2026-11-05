"""API and background indexing tests, on real Postgres and real Chroma in a temporary folder."""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Repo
from app.ingest import CloneError, remove_clone
from app.main import app
from app.vectorstore import _get_client, delete_collection, get_vectorstore
from tests.conftest import FakeGemini

pytestmark = pytest.mark.usefixtures("fresh_settings")

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"
CLICK_URL = "https://github.com/pallets/click"
client = TestClient(app)


@pytest.fixture(autouse=True)
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


def add_repo(url: str = CLICK_URL) -> dict:
    """Add a repo through the API. TestClient finishes background tasks before returning."""
    response = client.post("/api/repos", json={"url": url})
    assert response.status_code == 201
    return client.get(f"/api/repos/{response.json()['id']}").json()


def use_small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the sample repo produce several small embedding batches."""
    monkeypatch.setenv("CHUNK_SIZE", "80")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    monkeypatch.setenv("EMBED_BATCH_SIZE", "2")


def collection_count(repo_id: int) -> int:
    """Count the vectors stored for a repo."""
    return get_vectorstore(repo_id)._collection.count()


# --- adding repos


def test_add_repo_indexes_it_in_the_background() -> None:
    repo = add_repo()

    assert repo["status"] == "ready"
    assert repo["error"] is None
    assert repo["name"] == "pallets/click"
    assert repo["file_count"] == 2
    assert repo["languages"] == {"Markdown": 1, "Python": 1}
    assert repo["chunk_count"] > 0
    assert repo["chunks_embedded"] == repo["chunk_count"]
    assert collection_count(repo["id"]) == repo["chunk_count"]


@pytest.mark.parametrize(
    "same_repo",
    [
        "https://github.com/pallets/click",
        "https://github.com/Pallets/Click",
        "https://github.com/PALLETS/CLICK/",
        "https://github.com/pallets/click.git",
        "  https://github.com/pallets/click  ",
    ],
)
def test_adding_the_same_repo_again_returns_the_existing_one(same_repo: str, cloned_urls: list[str]) -> None:
    first = add_repo()

    response = client.post("/api/repos", json={"url": same_repo})

    assert response.status_code == 200
    assert response.json()["id"] == first["id"]
    assert cloned_urls == [CLICK_URL]


def test_repo_url_is_stored_in_normalized_form() -> None:
    repo = add_repo("https://github.com/Pallets/Click.git")

    assert repo["url"] == CLICK_URL
    assert repo["name"] == "pallets/click"


@pytest.mark.parametrize("url", ["not a url", "https://gitlab.com/a/b", "https://github.com/onlyowner"])
def test_add_repo_rejects_invalid_urls(url: str) -> None:
    response = client.post("/api/repos", json={"url": url})

    assert response.status_code == 422
    assert "github.com/owner/repo" in response.json()["detail"]


# --- reading repos


def test_list_and_get_repos() -> None:
    first = add_repo("https://github.com/a/one")
    second = add_repo("https://github.com/a/two")

    assert [repo["id"] for repo in client.get("/api/repos").json()] == [second["id"], first["id"]]
    assert client.get(f"/api/repos/{first['id']}").json()["name"] == "a/one"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/repos/999"),
        ("post", "/api/repos/999/reindex"),
        ("delete", "/api/repos/999"),
        ("get", "/api/repos/999/file?path=README.md"),
    ],
)
def test_unknown_repo_returns_404(method: str, path: str) -> None:
    assert client.request(method, path).status_code == 404


# --- failures


def test_failed_clone_is_saved_as_a_readable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(url: str, repo_id: int) -> Path:
        raise CloneError("Could not clone that repo.")

    monkeypatch.setattr("app.ingest.clone_repo", fail)

    repo = add_repo()

    assert repo["status"] == "failed"
    assert repo["error"] == "Could not clone that repo."


def test_repo_without_source_files_fails_clearly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "logo.png").write_bytes(b"not source")
    monkeypatch.setattr("app.ingest.clone_repo", lambda url, repo_id: tmp_path)

    repo = add_repo()

    assert repo["status"] == "failed"
    assert "no source files" in repo["error"]


def test_unexpected_errors_are_not_shown_to_users(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(url: str, repo_id: int) -> Path:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr("app.ingest.clone_repo", fail)

    repo = add_repo()

    assert repo["status"] == "failed"
    assert "secret internal detail" not in repo["error"]


def test_max_files_limits_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILES", "1")

    assert add_repo()["file_count"] == 1


# --- embedding batches, throttling and rate limits


def test_chunks_are_embedded_in_small_throttled_batches(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_small_chunks(monkeypatch)
    monkeypatch.setenv("EMBED_DELAY_SECONDS", "1.5")

    repo = add_repo()

    assert repo["status"] == "ready"
    assert repo["chunk_count"] > 2
    assert sum(fake_gemini.batch_sizes) == repo["chunk_count"]
    assert max(fake_gemini.batch_sizes) == 2
    # One pause between each pair of batches, and none after the last.
    assert fake_gemini.sleeps == [1.5] * (len(fake_gemini.batch_sizes) - 1)
    assert collection_count(repo["id"]) == repo["chunk_count"]


def test_rate_limit_waits_for_the_delay_gemini_gives_then_continues(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_small_chunks(monkeypatch)
    monkeypatch.setenv("MAX_RETRIES", "0")
    fake_gemini.statuses = [200, 429, 200]
    fake_gemini.error_details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}]

    repo = add_repo()

    assert repo["status"] == "ready"
    assert repo["chunks_embedded"] == repo["chunk_count"]
    assert 37.0 in fake_gemini.sleeps
    assert collection_count(repo["id"]) == repo["chunk_count"]


def test_rate_limit_without_a_delay_waits_for_the_default_window(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "0")
    monkeypatch.setenv("EMBED_RATE_LIMIT_WAIT_SECONDS", "45")
    fake_gemini.statuses = [429, 200]

    repo = add_repo()

    assert repo["status"] == "ready"
    assert 45.0 in fake_gemini.sleeps


def test_temporary_server_errors_also_wait(fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_RETRIES", "0")
    fake_gemini.statuses = [503, 200]

    assert add_repo()["status"] == "ready"


def test_endless_rate_limiting_fails_and_keeps_progress(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_small_chunks(monkeypatch)
    monkeypatch.setenv("MAX_RETRIES", "0")
    monkeypatch.setenv("EMBED_MAX_WAIT_SECONDS", "100")
    fake_gemini.statuses = [200, 429]
    fake_gemini.error_details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]

    repo = add_repo()

    assert repo["status"] == "failed"
    assert "kept refusing" in repo["error"]
    assert repo["chunks_embedded"] == 2
    assert repo["chunk_count"] > 2
    assert fake_gemini.sleeps.count(30.0) == 3


def test_a_retry_delay_longer_than_the_limit_fails_without_waiting(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "0")
    fake_gemini.statuses = [429]
    fake_gemini.error_details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "90000s"}]

    assert add_repo()["status"] == "failed"
    assert 90000.0 not in fake_gemini.sleeps


def test_daily_quota_fails_straight_away_with_a_clear_message(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "0")
    fake_gemini.statuses = [429]
    fake_gemini.error_details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "EmbedContentRequestsPerDayPerProjectPerModel-FreeTier"}],
        },
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "5s"},
    ]

    repo = add_repo()

    assert repo["status"] == "failed"
    assert "daily" in repo["error"]
    assert fake_gemini.sleeps == []


def test_permanent_gemini_errors_fail_with_a_friendly_message(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "0")
    fake_gemini.statuses = [404]

    repo = add_repo()

    assert repo["status"] == "failed"
    assert "can't find that model" in repo["error"]
    assert fake_gemini.attempts == 1


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Fake local embeddings that record the size of each batch."""

    batch_sizes: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return super().embed_documents(texts)


def test_local_embeddings_skip_pauses_and_still_report_progress(
    monkeypatch: pytest.MonkeyPatch, recorded_sleeps: list[float]
) -> None:
    use_small_chunks(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBED_DELAY_SECONDS", "5")
    monkeypatch.setattr("app.ingest.LOCAL_EMBED_BATCH_SIZE", 3)
    embeddings = CountingEmbeddings(size=8, batch_sizes=[])
    monkeypatch.setattr("app.vectorstore.get_embeddings", lambda: embeddings)

    repo = add_repo()

    assert repo["status"] == "ready"
    assert repo["chunks_embedded"] == repo["chunk_count"]
    assert sum(embeddings.batch_sizes) == repo["chunk_count"]
    assert len(embeddings.batch_sizes) > 1 and max(embeddings.batch_sizes) == 3
    assert recorded_sleeps == []


def test_local_embedding_errors_fail_without_rate_limit_waits(
    monkeypatch: pytest.MonkeyPatch, recorded_sleeps: list[float]
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")

    def fail() -> None:
        raise ValueError("EMBEDDING_MODEL 'bad' isn't a FastEmbed model.")

    monkeypatch.setattr("app.vectorstore.get_embeddings", fail)

    repo = add_repo()

    assert repo["status"] == "failed"
    assert "isn't a FastEmbed model" in repo["error"]
    assert recorded_sleeps == []


def test_missing_gemini_setting_is_a_readable_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model-name")

    repo = add_repo()

    assert repo["status"] == "failed"
    assert "GOOGLE_API_KEY" in repo["error"]


# --- reindex and delete


def test_reindex_rebuilds_the_collection_without_duplicates(cloned_urls: list[str]) -> None:
    repo = add_repo()

    response = client.post(f"/api/repos/{repo['id']}/reindex")

    assert response.status_code == 202
    again = client.get(f"/api/repos/{repo['id']}").json()
    assert again["status"] == "ready"
    assert again["chunks_embedded"] == again["chunk_count"]
    assert collection_count(repo["id"]) == repo["chunk_count"]
    assert cloned_urls == [CLICK_URL, CLICK_URL]


def test_failed_repo_can_be_reindexed(monkeypatch: pytest.MonkeyPatch, cloned_urls: list[str]) -> None:
    with monkeypatch.context() as patch:
        patch.setattr("app.ingest.clone_repo", lambda url, repo_id: (_ for _ in ()).throw(CloneError("nope")))
        repo = add_repo()
    assert repo["status"] == "failed"

    client.post(f"/api/repos/{repo['id']}/reindex")

    again = client.get(f"/api/repos/{repo['id']}").json()
    assert again["status"] == "ready"
    assert again["error"] is None


@pytest.mark.parametrize("busy_status", ["pending", "cloning", "indexing"])
@pytest.mark.parametrize(("method", "suffix"), [("post", "/reindex"), ("delete", "")])
def test_busy_repo_cannot_be_reindexed_or_deleted(
    db_session: Session, busy_status: str, method: str, suffix: str
) -> None:
    repo = Repo(url=CLICK_URL, name="pallets/click", status=busy_status)
    db_session.add(repo)
    db_session.commit()

    response = client.request(method, f"/api/repos/{repo.id}{suffix}")

    assert response.status_code == 409
    assert busy_status in response.json()["detail"]


def test_delete_removes_the_row_clone_and_vectors() -> None:
    repo = add_repo()
    clone = get_settings().repos_dir / str(repo["id"])
    assert clone.is_dir()

    response = client.delete(f"/api/repos/{repo['id']}")

    assert response.status_code == 204
    assert client.get(f"/api/repos/{repo['id']}").status_code == 404
    assert not clone.exists()
    assert f"repo_{repo['id']}" not in [collection.name for collection in _get_client().list_collections()]


def test_delete_collection_ignores_a_missing_collection() -> None:
    delete_collection(987654)


# --- vector store


def test_collection_uses_cosine_similarity() -> None:
    repo = add_repo()

    collection = _get_client().get_collection(f"repo_{repo['id']}")

    assert collection.configuration["hnsw"]["space"] == "cosine"


def test_stored_chunks_keep_their_citation_metadata() -> None:
    repo = add_repo()

    stored = get_vectorstore(repo["id"]).similarity_search("greet", k=repo["chunk_count"])

    python_chunk = next(document for document in stored if document.metadata["path"] == "src/app.py")
    assert python_chunk.page_content.startswith("# File: src/app.py\n")
    assert python_chunk.metadata["language"] == "Python"
    assert python_chunk.metadata["start_line"] == 1


# --- reading files


def test_read_file_returns_its_content() -> None:
    repo = add_repo()

    response = client.get(f"/api/repos/{repo['id']}/file", params={"path": "src/app.py"})

    assert response.status_code == 200
    assert response.json()["path"] == "src/app.py"
    assert "def greet" in response.json()["content"]


def test_path_traversal_cannot_reach_a_real_secret_file() -> None:
    repo = add_repo()
    # The clone is at <root>/data/repos/<id>, so this path really does lead to <root>/.env.
    secret = get_settings().data_dir.parent / ".env"
    secret.write_text("GOOGLE_API_KEY=secret", encoding="utf-8")
    clone = get_settings().repos_dir / str(repo["id"])
    assert (clone / "../../../.env").resolve() == secret.resolve()

    response = client.get(f"/api/repos/{repo['id']}/file", params={"path": "../../../.env"})

    assert response.status_code == 404
    assert "secret" not in response.text


@pytest.mark.parametrize(
    "path",
    [
        "../../../.env",
        "..\\..\\..\\.env",
        "src/../../../../.env",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "does_not_exist.py",
        "src",
        "bad\x00name.py",
    ],
)
def test_read_file_rejects_bad_paths(path: str) -> None:
    repo = add_repo()

    assert client.get(f"/api/repos/{repo['id']}/file", params={"path": path}).status_code == 404


def test_read_file_hides_the_git_folder() -> None:
    repo = add_repo()
    clone = get_settings().repos_dir / str(repo["id"])
    (clone / ".git").mkdir()
    (clone / ".git" / "config").write_text("[core]", encoding="utf-8")

    assert client.get(f"/api/repos/{repo['id']}/file", params={"path": ".git/config"}).status_code == 404


def test_read_file_blocks_a_link_leading_outside_the_clone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Creating real symlinks needs special rights on Windows, so this makes one path resolve elsewhere.
    repo = add_repo()
    outside = tmp_path / "secret.py"
    outside.write_text("secret", encoding="utf-8")
    real_resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path: outside if path.name == "link.py" else real_resolve(path))

    assert client.get(f"/api/repos/{repo['id']}/file", params={"path": "link.py"}).status_code == 404


def test_read_file_rejects_large_and_binary_files(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = add_repo()
    clone = get_settings().repos_dir / str(repo["id"])
    (clone / "binary.py").write_bytes(b"\xff\xfe\x00\x01")
    (clone / "large.py").write_text("x" * 50, encoding="utf-8")
    monkeypatch.setenv("MAX_FILE_BYTES", "10")
    get_settings.cache_clear()

    assert client.get(f"/api/repos/{repo['id']}/file", params={"path": "large.py"}).status_code == 413
    assert client.get(f"/api/repos/{repo['id']}/file", params={"path": "binary.py"}).status_code == 415


# --- startup


def test_startup_marks_interrupted_repos_as_failed(db_session: Session) -> None:
    for number, status in enumerate(["pending", "cloning", "indexing", "ready", "failed"]):
        db_session.add(Repo(url=f"https://github.com/a/{number}", name=f"a/{number}", status=status))
    db_session.commit()

    # Using TestClient as a context manager runs the app's startup code.
    with TestClient(app):
        pass

    db_session.expire_all()
    repos = db_session.scalars(select(Repo).order_by(Repo.id)).all()
    assert [repo.status for repo in repos] == ["failed", "failed", "failed", "ready", "failed"]
    assert all("server restarted" in repo.error for repo in repos[:3])
    assert repos[3].error is None
    assert repos[4].error is None


def test_app_still_starts_when_the_database_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise OperationalError("SELECT", {}, Exception("database down"))

    monkeypatch.setattr("app.main.fail_interrupted_repos", fail)

    with TestClient(app) as started:
        assert started.get("/api/health").status_code == 200
