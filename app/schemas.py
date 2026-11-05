"""Pydantic request and response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RepoCreate(BaseModel):
    """Request to add a GitHub repo."""

    url: str


class RepoOut(BaseModel):
    """A repo and its indexing status. Progress is chunks_embedded out of chunk_count."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    name: str
    status: str
    error: str | None
    file_count: int
    chunk_count: int
    chunks_embedded: int
    embedding_provider: str | None
    embedding_model: str | None
    languages: dict[str, int]
    created_at: datetime


class RepoFileOut(BaseModel):
    """The text of one file in a cloned repo."""

    path: str
    content: str


class AskRequest(BaseModel):
    """A question about a repo. Leave out conversation_id to start a new conversation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None


class ConversationOut(BaseModel):
    """A chat about a repo."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    repo_id: int
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    """One message in a conversation. AI messages list the sources they cite."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str
    sources: list[dict[str, Any]]
    created_at: datetime
