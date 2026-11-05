"""Tests for the onboarding guide, with fake models, real Postgres and real Chroma."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Repo
from app.llm import FAKE_CHAT_RESPONSES
from app.main import app
from app.rag import MAX_README_CHARS, ONBOARDING_PROBES, _probe_excerpts, _read_readme
from tests.conftest import FakeGemini, RecordingChatModel

pytestmark = pytest.mark.usefixtures("fresh_settings", "cloned_urls")

client = TestClient(app)
GUIDE = FAKE_CHAT_RESPONSES[0]


@pytest.fixture
def repo_id() -> int:
    """Add and index the sample repo."""
    return client.post("/api/repos", json={"url": "https://github.com/acme/sample"}).json()["id"]


def system_prompt(model: RecordingChatModel) -> str:
    """Return the system message of the guide request."""
    return model.prompts[0][0].content


# --- generating and reading


def test_post_generates_and_saves_and_get_returns_the_saved_guide(repo_id: int, db_session: Session) -> None:
    created = client.post(f"/api/repos/{repo_id}/guide")

    assert created.status_code == 200
    assert created.json() == {"repo_id": repo_id, "guide": GUIDE}
    assert db_session.get(Repo, repo_id).onboarding_guide == GUIDE
    assert client.get(f"/api/repos/{repo_id}/guide").json() == created.json()


def test_get_does_not_call_the_model(repo_id: int, recording_model: RecordingChatModel) -> None:
    client.post(f"/api/repos/{repo_id}/guide")
    recording_model.prompts.clear()

    assert client.get(f"/api/repos/{repo_id}/guide").status_code == 200
    assert recording_model.prompts == []


def test_get_before_generating_returns_404(repo_id: int) -> None:
    response = client.get(f"/api/repos/{repo_id}/guide")

    assert response.status_code == 404
    assert "No guide yet" in response.json()["detail"]


def test_post_replaces_an_earlier_guide(repo_id: int, recording_model: RecordingChatModel) -> None:
    client.post(f"/api/repos/{repo_id}/guide")
    recording_model.responses = ["A newer guide."]
    recording_model.i = 0

    client.post(f"/api/repos/{repo_id}/guide")

    assert client.get(f"/api/repos/{repo_id}/guide").json()["guide"] == "A newer guide."


def test_reindexing_clears_the_old_guide(repo_id: int) -> None:
    client.post(f"/api/repos/{repo_id}/guide")

    client.post(f"/api/repos/{repo_id}/reindex")

    assert client.get(f"/api/repos/{repo_id}/guide").status_code == 404


# --- what the model receives


def test_prompt_has_the_readme_tree_languages_and_excerpts(repo_id: int, recording_model: RecordingChatModel) -> None:
    client.post(f"/api/repos/{repo_id}/guide")

    prompt = system_prompt(recording_model)
    assert "A tiny project used by the RepoGuide tests." in prompt  # README
    assert "src/\n  app.py" in prompt  # file tree
    assert "Markdown (1)" in prompt and "Python (1)" in prompt  # language stats
    assert "[1] " in prompt and "def greet" in prompt  # excerpts
    assert "acme/sample" in recording_model.prompts[0][-1].content


def test_prompt_asks_for_every_section_and_forbids_invented_files(
    repo_id: int, recording_model: RecordingChatModel
) -> None:
    client.post(f"/api/repos/{repo_id}/guide")

    prompt = system_prompt(recording_model)
    for section in [
        "What it does",
        "Tech stack",
        "Architecture",
        "Directory guide",
        "Where to start reading",
        "How to run locally",
        "Good first tasks",
    ]:
        assert section in prompt
    assert "Only mention a file or folder if it appears" in prompt
    assert "untrusted" in prompt


def test_probes_cover_each_topic_and_duplicates_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    searches: list[tuple[str, int]] = []

    class StubStore:
        def similarity_search(self, query: str, k: int) -> list[Document]:
            searches.append((query, k))
            shared = {"path": "shared.py", "start_line": 1}
            own = {"path": f"{len(searches)}.py", "start_line": 1}
            return [Document(page_content="shared", metadata=shared), Document(page_content="own", metadata=own)]

    monkeypatch.setattr("app.rag.get_vectorstore", lambda repo: StubStore())

    documents = _probe_excerpts(1)

    assert searches == [(probe, 3) for probe in ONBOARDING_PROBES]
    assert len(ONBOARDING_PROBES) == 5
    paths = [document.metadata["path"] for document in documents]
    assert paths.count("shared.py") == 1
    assert len(paths) == 6


# --- reading the README


def test_readme_is_cut_to_the_limit(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("x" * (MAX_README_CHARS + 500), encoding="utf-8")

    assert len(_read_readme(tmp_path)) == MAX_README_CHARS


@pytest.mark.parametrize("name", ["README.md", "readme.rst", "Readme.txt", "README"])
def test_readme_is_found_by_common_names(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("hello", encoding="utf-8")

    assert _read_readme(tmp_path) == "hello"


def test_missing_readme_is_stated_in_the_prompt(repo_id: int, recording_model: RecordingChatModel) -> None:
    (get_settings().repos_dir / str(repo_id) / "README.md").unlink()

    client.post(f"/api/repos/{repo_id}/guide")

    assert "(this repo has no README)" in system_prompt(recording_model)


def test_a_linked_readme_is_not_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Creating real symlinks needs special rights on Windows, so this pretends the file is a link.
    (tmp_path / "README.md").write_text("secret from outside the clone", encoding="utf-8")
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.name == "README.md")

    assert _read_readme(tmp_path) == ""


def test_a_readme_that_is_not_utf8_does_not_fail(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_bytes(b"caf\xe9")

    assert _read_readme(tmp_path).startswith("caf")


# --- errors


def test_unknown_repo_returns_404() -> None:
    assert client.get("/api/repos/999/guide").status_code == 404
    assert client.post("/api/repos/999/guide").status_code == 404


def test_repo_that_is_not_ready_returns_409(db_session: Session) -> None:
    repo = Repo(url="https://github.com/a/b", name="a/b", status="indexing")
    db_session.add(repo)
    db_session.commit()

    assert client.post(f"/api/repos/{repo.id}/guide").status_code == 409


def test_index_from_another_embedding_model_returns_409(repo_id: int, db_session: Session) -> None:
    repo = db_session.get(Repo, repo_id)
    repo.embedding_model = "other"
    db_session.commit()

    response = client.post(f"/api/repos/{repo_id}/guide")

    assert response.status_code == 409
    assert response.json()["detail"].startswith("Re-index needed")


def test_gemini_error_returns_503_and_keeps_the_old_guide(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, db_session: Session
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "0")
    repo_id = client.post("/api/repos", json={"url": "https://github.com/acme/sample"}).json()["id"]
    repo = db_session.get(Repo, repo_id)
    repo.onboarding_guide = "The earlier guide."
    db_session.commit()
    fake_gemini.statuses_by_model["guide-model-name"] = 503

    response = client.post(f"/api/repos/{repo_id}/guide")

    assert fake_gemini.models_called()[-1] == "guide-model-name"
    assert response.status_code == 503
    assert "Every configured model" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.get(Repo, repo_id).onboarding_guide == "The earlier guide."


def test_guides_are_written_by_the_guide_chain(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GUIDE_MODELS", "first-guide-model,second-guide-model")
    repo_id = client.post("/api/repos", json={"url": "https://github.com/acme/sample"}).json()["id"]
    fake_gemini.requests.clear()
    fake_gemini.statuses_by_model["first-guide-model"] = 503

    response = client.post(f"/api/repos/{repo_id}/guide")

    assert response.status_code == 200
    assert response.json()["guide"] == "hello"
    assert [model for model in fake_gemini.models_called() if "embedding" not in model] == [
        "first-guide-model",
        "second-guide-model",
    ]


def test_missing_chat_setting_returns_503(repo_id: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GUIDE_MODELS", "guide-model-name")
    get_settings.cache_clear()

    response = client.post(f"/api/repos/{repo_id}/guide")

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]
