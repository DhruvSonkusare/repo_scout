"""Database engine, session factory, ORM models and the get_db dependency."""

from collections.abc import Iterator
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, MetaData, String, Text, create_engine, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import get_settings

# create_engine doesn't connect yet: it opens connections when they're first needed.
# On Windows, psycopg waits out the timeout even when the connection is refused, and
# its default is 130s per address, so keep it short to fail fast when Postgres is down.
engine = create_engine(
    get_settings().database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    # Predictable constraint names let later migrations alter or drop them by name.
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


# A repo in one of these states has a background job queued or running.
IN_PROGRESS_STATUSES = ("pending", "cloning", "indexing")


class Repo(Base):
    """A GitHub repository added for indexing."""

    __tablename__ = "repos"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending", server_default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    file_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    chunks_embedded: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # What built this repo's vectors. Searching needs the same model. NULL means unknown.
    embedding_provider: Mapped[str | None] = mapped_column(Text)
    embedding_model: Mapped[str | None] = mapped_column(Text)
    languages: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    onboarding_guide: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # passive_deletes leaves child rows to the database's ON DELETE CASCADE,
    # so SQLAlchemy doesn't load them all just to delete them.
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan", passive_deletes=True
    )


class Conversation(Base):
    """A chat about one repo."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repos.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    repo: Mapped[Repo] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True
    )


class Message(Base):
    """One user or AI message in a conversation."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(10))
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))
    # The chat model that wrote an AI message. NULL for user messages and older answers.
    model: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Search(Base):
    """A Repo Finder search for one problem statement."""

    __tablename__ = "searches"

    id: Mapped[int] = mapped_column(primary_key=True)
    problem: Mapped[str] = mapped_column(Text)
    plan: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Best match first, then repos the model never assessed. The id keeps equal scores in saved order.
    results: Mapped[list["SearchResult"]] = relationship(
        back_populates="search",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="(SearchResult.score.desc().nulls_last(), SearchResult.id)",
    )


class SearchResult(Base):
    """One ranked repo returned by a search."""

    __tablename__ = "search_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    search_id: Mapped[int] = mapped_column(ForeignKey("searches.id", ondelete="CASCADE"), index=True)
    full_name: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    stars: Mapped[int] = mapped_column(Integer)
    language: Mapped[str | None] = mapped_column(Text)
    last_pushed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # NULL means the model never assessed this repo.
    score: Mapped[int | None] = mapped_column(Integer)
    why_it_fits: Mapped[str] = mapped_column(Text)
    caveats: Mapped[str] = mapped_column(Text)
    how_to_start: Mapped[list[str]] = mapped_column(JSONB)
    # SET NULL keeps the search result when its indexed repo is deleted.
    repo_id: Mapped[int | None] = mapped_column(ForeignKey("repos.id", ondelete="SET NULL"), index=True)
    problem_guide: Mapped[str | None] = mapped_column(Text)
    # How the guide was checked against the code, with its draft, so the two can be compared.
    guide_verification: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    search: Mapped[Search] = relationship(back_populates="results")
    repo: Mapped[Repo | None] = relationship()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed afterwards."""
    with SessionLocal() as session:
        yield session
