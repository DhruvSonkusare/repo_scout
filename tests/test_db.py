"""Tests for the ORM models and migrations against real Postgres."""

from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import Conversation, Message, Repo, Search, SearchResult, get_db


def make_search_result(search: Search, repo: Repo | None = None) -> SearchResult:
    """Build a search result with placeholder values."""
    return SearchResult(
        search=search,
        repo=repo,
        full_name="pallets/click",
        url="https://github.com/pallets/click",
        description="Command line toolkit",
        stars=15000,
        language="Python",
        last_pushed=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        score=90,
        why_it_fits="Builds command line tools.",
        caveats="Python only.",
        how_to_start=["pip install click", "Read the quickstart"],
    )


def make_repo(url: str = "https://github.com/pallets/click") -> Repo:
    """Build a repo with placeholder values."""
    return Repo(url=url, name="pallets/click")


def test_insert_one_row_into_each_table(db_session: Session) -> None:
    repo = make_repo()
    repo.languages = {"Python": 120}
    conversation = Conversation(repo=repo, title="How do options work?")
    message = Message(
        conversation=conversation,
        role="ai",
        content="See parser.py [1].",
        sources=[{"path": "src/click/parser.py", "start_line": 1, "end_line": 40}],
    )
    search = Search(problem="Build a CLI in Python", plan={"queries": ["python cli"], "language": "Python"})
    result = make_search_result(search, repo)
    db_session.add_all([repo, conversation, message, search, result])
    db_session.commit()
    db_session.expire_all()

    assert db_session.get(Repo, repo.id).languages == {"Python": 120}
    assert db_session.get(Conversation, conversation.id).repo_id == repo.id
    assert db_session.get(Message, message.id).sources[0]["path"] == "src/click/parser.py"
    assert db_session.get(Search, search.id).plan["language"] == "Python"
    assert db_session.get(SearchResult, result.id).repo_id == repo.id
    assert db_session.get(SearchResult, result.id).how_to_start == ["pip install click", "Read the quickstart"]


def test_repo_defaults_come_from_the_database(db_session: Session) -> None:
    db_session.execute(text("INSERT INTO repos (url, name) VALUES ('https://github.com/a/b', 'a/b')"))
    db_session.commit()

    repo = db_session.scalars(select(Repo)).one()

    assert repo.status == "pending"
    assert repo.file_count == 0
    assert repo.chunk_count == 0
    assert repo.languages == {}


def test_created_at_is_timezone_aware(db_session: Session) -> None:
    repo = make_repo()
    db_session.add(repo)
    db_session.commit()
    db_session.refresh(repo)

    assert repo.created_at.tzinfo is not None


def test_json_columns_are_queryable_as_jsonb(db_session: Session) -> None:
    python_repo = make_repo("https://github.com/a/python")
    python_repo.languages = {"Python": 10}
    go_repo = make_repo("https://github.com/a/go")
    go_repo.languages = {"Go": 10}
    db_session.add_all([python_repo, go_repo])
    db_session.commit()

    # @> (contains) exists for JSONB but not for plain JSON or SQLite.
    matches = db_session.scalars(select(Repo).where(Repo.languages.op("@>")({"Python": 10}))).all()

    assert [repo.id for repo in matches] == [python_repo.id]
    column_type = db_session.scalar(
        text("SELECT data_type FROM information_schema.columns WHERE table_name = 'repos' AND column_name = 'languages'")
    )
    assert column_type == "jsonb"


def test_repo_url_must_be_unique(db_session: Session) -> None:
    db_session.add(make_repo())
    db_session.commit()

    db_session.add(make_repo())
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_deleting_repo_cascades_to_chats_and_keeps_search_result(db_session: Session) -> None:
    repo = make_repo()
    conversation = Conversation(repo=repo, title="Question")
    message = Message(conversation=conversation, role="user", content="Hello?")
    search = Search(problem="Build a CLI", plan={})
    result = make_search_result(search, repo)
    db_session.add_all([repo, conversation, message, search, result])
    db_session.commit()

    db_session.delete(repo)
    db_session.commit()
    db_session.expire_all()

    assert db_session.scalar(select(func.count()).select_from(Conversation)) == 0
    assert db_session.scalar(select(func.count()).select_from(Message)) == 0
    assert db_session.get(SearchResult, result.id).repo_id is None


def test_foreign_keys_cascade_in_the_database_itself(db_session: Session) -> None:
    repo = make_repo()
    conversation = Conversation(repo=repo, title="Question")
    db_session.add_all([repo, conversation, Message(conversation=conversation, role="user", content="Hello?")])
    db_session.commit()

    # Raw SQL skips the ORM, so only ondelete="CASCADE" in the schema can remove the children.
    db_session.execute(text("DELETE FROM repos"))
    db_session.commit()

    assert db_session.scalar(select(func.count()).select_from(Message)) == 0


def test_deleting_search_removes_its_results(db_session: Session) -> None:
    search = Search(problem="Build a CLI", plan={})
    db_session.add_all([search, make_search_result(search)])
    db_session.commit()

    db_session.delete(search)
    db_session.commit()

    assert db_session.scalar(select(func.count()).select_from(SearchResult)) == 0


def test_get_db_yields_a_working_session() -> None:
    dependency = get_db()

    session = next(dependency)

    assert session.scalar(select(1)) == 1
    dependency.close()


def test_migrations_match_the_models(alembic_config: Config) -> None:
    # Fails if someone changes a model without generating a migration.
    command.check(alembic_config)
