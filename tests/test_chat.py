"""Tests for streamed answers and conversations, with fake models, real Postgres and real Chroma."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from sqlalchemy.orm import Session

from app.db import Conversation, Message, Repo
from app.llm import FAKE_CHAT_RESPONSES
from app.main import app
from app.rag import format_docs, retrieve
from tests.conftest import FakeGemini, RecordingChatModel

pytestmark = pytest.mark.usefixtures("fresh_settings", "cloned_urls")

client = TestClient(app)
ANSWER = FAKE_CHAT_RESPONSES[0]


@pytest.fixture
def repo_id() -> int:
    """Add and index the sample repo."""
    response = client.post("/api/repos", json={"url": "https://github.com/acme/sample"})
    assert response.status_code == 201
    return response.json()["id"]


def ask(repo_id: int, question: str = "What does greet do?", conversation_id: int | None = None) -> list[dict]:
    """Ask a question and parse every NDJSON line of the response."""
    body: dict[str, Any] = {"question": question}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    response = client.post(f"/api/repos/{repo_id}/ask", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines()]


def answer_text(events: list[dict]) -> str:
    """Join the token events into the full answer."""
    return "".join(event["text"] for event in events if event["type"] == "token")


# --- streaming


def test_ask_streams_meta_then_tokens_then_done(repo_id: int) -> None:
    events = ask(repo_id)

    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "done"
    assert all(event["type"] == "token" for event in events[1:-1])
    assert len(events) > 3
    assert answer_text(events) == ANSWER


def test_meta_lists_numbered_sources_with_line_numbers(repo_id: int) -> None:
    sources = ask(repo_id)[0]["sources"]

    assert [source["n"] for source in sources] == list(range(1, len(sources) + 1))
    assert {source["path"] for source in sources} == {"README.md", "src/app.py"}
    assert all(source["start_line"] >= 1 and source["end_line"] >= source["start_line"] for source in sources)


def test_ask_saves_both_messages(repo_id: int, db_session: Session) -> None:
    events = ask(repo_id, "What does greet do?")

    conversation_id = events[0]["conversation_id"]
    messages = client.get(f"/api/conversations/{conversation_id}/messages").json()
    assert [(message["role"], message["content"]) for message in messages] == [
        ("user", "What does greet do?"),
        ("ai", ANSWER),
    ]
    assert messages[1]["id"] == events[-1]["message_id"]
    assert messages[1]["sources"] == events[0]["sources"]
    assert messages[0]["sources"] == []


def test_empty_tokens_are_not_sent(repo_id: int, monkeypatch: pytest.MonkeyPatch) -> None:
    class ModelWithEmptyChunks(FakeListChatModel):
        async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
            for text in ["", "Hello", "", " world", ""]:
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    monkeypatch.setattr("app.rag.get_chat_model", lambda: ModelWithEmptyChunks(responses=["unused"]))

    events = ask(repo_id)

    tokens = [event["text"] for event in events if event["type"] == "token"]
    assert tokens == ["Hello", " world"]
    assert events[-1]["type"] == "done"


def test_first_question_skips_the_condense_call(repo_id: int, recording_model: RecordingChatModel) -> None:
    ask(repo_id, "What does greet do?")

    assert len(recording_model.prompts) == 1
    assert "standalone" not in recording_model.prompts[0][0].content


def test_excerpts_and_question_reach_the_model(repo_id: int, recording_model: RecordingChatModel) -> None:
    ask(repo_id, "What does greet do?")

    system, question = recording_model.prompts[0][0], recording_model.prompts[0][-1]
    assert "[1] " in system.content
    assert "def greet" in system.content
    assert "untrusted" in system.content
    assert question.content == "What does greet do?"


# --- follow-ups


def test_follow_up_continues_the_same_conversation(repo_id: int, db_session: Session) -> None:
    first = ask(repo_id, "What does greet do?")
    conversation_id = first[0]["conversation_id"]

    second = ask(repo_id, "Where is that tested?", conversation_id)

    assert second[0]["conversation_id"] == conversation_id
    assert len(client.get(f"/api/conversations/{conversation_id}/messages").json()) == 4
    assert len(client.get(f"/api/repos/{repo_id}/conversations").json()) == 1


def test_follow_up_searches_with_the_condensed_question(
    repo_id: int, monkeypatch: pytest.MonkeyPatch, recording_model: RecordingChatModel
) -> None:
    conversation_id = ask(repo_id, "What does greet do?")[0]["conversation_id"]
    recording_model.prompts.clear()
    recording_model.responses = ["Where is greet tested?", ANSWER]
    recording_model.i = 0
    searched: list[str] = []

    async def fake_retrieve(repo: int, question: str) -> list[Document]:
        searched.append(question)
        return []

    monkeypatch.setattr("app.rag.retrieve", fake_retrieve)

    events = ask(repo_id, "Where is that tested?", conversation_id)

    assert searched == ["Where is greet tested?"]
    assert answer_text(events) == ANSWER
    condense_prompt = recording_model.prompts[0]
    assert "standalone" in condense_prompt[0].content
    assert [message.content for message in condense_prompt[1:]] == [
        "What does greet do?",
        ANSWER,
        "Where is that tested?",
    ]
    # The answer is written for the user's own words, with the history for context.
    assert recording_model.prompts[1][-1].content == "Where is that tested?"


def test_only_the_last_six_turns_are_used_as_history(
    repo_id: int, db_session: Session, recording_model: RecordingChatModel
) -> None:
    conversation_id = ask(repo_id, "First question")[0]["conversation_id"]
    for number in range(8):
        db_session.add(Message(conversation_id=conversation_id, role="user", content=f"question {number}"))
        db_session.add(Message(conversation_id=conversation_id, role="ai", content=f"answer {number}"))
    db_session.commit()
    recording_model.prompts.clear()

    ask(repo_id, "Latest question", conversation_id)

    history = [message.content for message in recording_model.prompts[-1][1:-1]]
    assert len(history) == 12
    assert history[0] == "question 2"
    assert history[-1] == "answer 7"


# --- HTTP errors


def test_unknown_repo_returns_404() -> None:
    assert client.post("/api/repos/999/ask", json={"question": "Hi?"}).status_code == 404


@pytest.mark.parametrize("status", ["pending", "cloning", "indexing", "failed"])
def test_repo_that_is_not_ready_returns_409(db_session: Session, status: str) -> None:
    repo = Repo(url="https://github.com/a/b", name="a/b", status=status)
    db_session.add(repo)
    db_session.commit()

    response = client.post(f"/api/repos/{repo.id}/ask", json={"question": "Hi?"})

    assert response.status_code == 409
    assert status in response.json()["detail"]


@pytest.mark.parametrize(
    ("provider", "model"),
    [("fake", "some-other-model"), ("local", "deterministic-256"), ("gemini", "models/gemini-embedding-001")],
)
def test_index_built_with_another_embedding_model_needs_a_reindex(
    repo_id: int, db_session: Session, provider: str, model: str
) -> None:
    repo = db_session.get(Repo, repo_id)
    repo.embedding_provider, repo.embedding_model = provider, model
    db_session.commit()

    response = client.post(f"/api/repos/{repo_id}/ask", json={"question": "Hi?"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail.startswith("Re-index needed")
    assert f"{provider} / {model}" in detail
    assert "fake / deterministic-256" in detail


def test_changing_the_embedding_setting_needs_a_reindex_and_reindexing_fixes_it(
    repo_id: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr("app.routers.chat.embedding_identity", lambda: ("local", "new-model"))
        response = client.post(f"/api/repos/{repo_id}/ask", json={"question": "Hi?"})
    assert response.status_code == 409
    assert response.json()["detail"].startswith("Re-index needed")

    assert client.post(f"/api/repos/{repo_id}/reindex").status_code == 202

    assert ask(repo_id)[-1]["type"] == "done"


def test_repo_indexed_before_this_was_recorded_needs_a_reindex(repo_id: int, db_session: Session) -> None:
    repo = db_session.get(Repo, repo_id)
    repo.embedding_provider = repo.embedding_model = None
    db_session.commit()

    response = client.post(f"/api/repos/{repo_id}/ask", json={"question": "Hi?"})

    assert response.status_code == 409
    assert "an unknown model" in response.json()["detail"]


def test_a_rejected_question_saves_nothing(repo_id: int, db_session: Session) -> None:
    repo = db_session.get(Repo, repo_id)
    repo.embedding_model = "other"
    db_session.commit()

    client.post(f"/api/repos/{repo_id}/ask", json={"question": "Hi?"})

    assert db_session.query(Conversation).count() == 0
    assert db_session.query(Message).count() == 0


@pytest.mark.parametrize("question", ["", "   ", "x" * 4001])
def test_invalid_questions_return_422(repo_id: int, question: str) -> None:
    assert client.post(f"/api/repos/{repo_id}/ask", json={"question": question}).status_code == 422


def test_unknown_conversation_returns_404(repo_id: int) -> None:
    response = client.post(f"/api/repos/{repo_id}/ask", json={"question": "Hi?", "conversation_id": 999})

    assert response.status_code == 404


def test_a_conversation_from_another_repo_returns_404(repo_id: int) -> None:
    other_repo_id = client.post("/api/repos", json={"url": "https://github.com/acme/other"}).json()["id"]
    conversation_id = ask(other_repo_id)[0]["conversation_id"]

    response = client.post(f"/api/repos/{repo_id}/ask", json={"question": "Hi?", "conversation_id": conversation_id})

    assert response.status_code == 404


# --- errors while streaming


def test_gemini_error_becomes_an_error_event_and_saves_no_answer(
    fake_gemini: FakeGemini, monkeypatch: pytest.MonkeyPatch, db_session: Session
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "0")
    repo = client.post("/api/repos", json={"url": "https://github.com/acme/sample"}).json()
    assert client.get(f"/api/repos/{repo['id']}").json()["status"] == "ready"
    fake_gemini.statuses = [503]
    fake_gemini.attempts = 0

    events = ask(repo["id"])

    assert events[-1]["type"] == "error"
    assert "Gemini is busy" in events[-1]["message"]
    assert [message.role for message in db_session.query(Message).all()] == ["user"]


def test_unexpected_error_hides_details_from_users(repo_id: int, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(repo: int, question: str) -> list[Document]:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr("app.rag.retrieve", fail)

    events = ask(repo_id)

    assert events[-1]["type"] == "error"
    assert "secret" not in json.dumps(events)


# --- history endpoints


def test_conversations_are_listed_newest_first_and_titled_from_the_question(repo_id: int) -> None:
    first = ask(repo_id, "First question")[0]["conversation_id"]
    second = ask(repo_id, "Second question " + "x" * 100)[0]["conversation_id"]

    conversations = client.get(f"/api/repos/{repo_id}/conversations").json()

    assert [conversation["id"] for conversation in conversations] == [second, first]
    assert conversations[1]["title"] == "First question"
    assert len(conversations[0]["title"]) == 80


def test_history_endpoints_return_404_for_unknown_ids() -> None:
    assert client.get("/api/repos/999/conversations").status_code == 404
    assert client.get("/api/conversations/999/messages").status_code == 404


def test_deleting_a_repo_deletes_its_conversations(repo_id: int, db_session: Session) -> None:
    ask(repo_id)

    assert client.delete(f"/api/repos/{repo_id}").status_code == 204

    assert db_session.query(Conversation).count() == 0
    assert db_session.query(Message).count() == 0


# --- retrieval


def test_retrieval_uses_mmr_tuned_towards_relevance(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: dict[str, Any] = {}

    class StubRetriever:
        async def ainvoke(self, question: str) -> list[Document]:
            return []

    class StubStore:
        def as_retriever(self, **kwargs: Any) -> StubRetriever:
            requested.update(kwargs)
            return StubRetriever()

    monkeypatch.setattr("app.rag.get_vectorstore", lambda repo: StubStore())

    assert asyncio.run(retrieve(1, "Hello?")) == []
    assert requested == {"search_type": "mmr", "search_kwargs": {"k": 6, "fetch_k": 24, "lambda_mult": 0.85}}


# --- format_docs


def test_format_docs_numbers_excerpts_and_removes_the_path_prefix() -> None:
    document = Document(
        page_content="# File: src/app.py\ndef greet(): ...",
        metadata={"path": "src/app.py", "start_line": 4, "end_line": 6, "language": "Python"},
    )

    text = format_docs([document])

    assert text == "[1] src/app.py (lines 4-6)\n````\ndef greet(): ...\n````"
