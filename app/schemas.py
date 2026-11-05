"""Pydantic request and response models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


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
    languages: dict[str, int]
    created_at: datetime


class RepoFileOut(BaseModel):
    """The text of one file in a cloned repo."""

    path: str
    content: str
