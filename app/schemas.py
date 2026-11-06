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


class GuideOut(BaseModel):
    """A repo's onboarding guide, in Markdown."""

    repo_id: int
    guide: str


class SearchRequest(BaseModel):
    """A problem to find open-source repos for, in plain English."""

    model_config = ConfigDict(str_strip_whitespace=True)

    problem: str = Field(min_length=3, max_length=2000)


class GuideVerificationOut(BaseModel):
    """What checking a guide against the repo's source code found.

    status is verified, failed (the check couldn't finish) or skipped (no source files to check).
    draft is the guide before the check, so the final guide can be compared with it.
    """

    model_config = ConfigDict(from_attributes=True)

    status: str
    draft: str
    changed: bool
    changes: list[str]
    files_checked: list[str]


class SearchResultOut(BaseModel):
    """One repo found for a problem, and how well it fits."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    url: str
    description: str | None
    stars: int
    language: str | None
    last_pushed: datetime | None
    score: int | None  # None means the model never assessed this repo
    why_it_fits: str
    caveats: str
    how_to_start: list[str]
    repo_id: int | None
    problem_guide: str | None
    guide_verification: GuideVerificationOut | None


class SearchSummaryOut(BaseModel):
    """A past search, without its results."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    problem: str
    created_at: datetime


class SearchOut(SearchSummaryOut):
    """A search with its plan and its results, best match first."""

    plan: dict[str, Any]
    results: list[SearchResultOut]


class ProblemGuideOut(BaseModel):
    """A guide to solving the user's problem with one repo, in Markdown, with what it was written for."""

    result_id: int
    problem: str
    repo_id: int
    repo_name: str
    guide: str
    verdict: str | None  # Fully, Partly or No, if the guide states one
    verification: GuideVerificationOut


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
    """One message in a conversation. AI messages list the sources they cite and the model that wrote them."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str
    sources: list[dict[str, Any]]
    model: str | None
    created_at: datetime
