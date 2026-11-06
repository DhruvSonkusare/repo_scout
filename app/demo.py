"""Limits for the public demo (DEMO_MODE=true). With demo mode off, everything here lets requests through.

Guests are limited per IP and share one daily cap. Signed-in users are limited per user, and have their
own totals, so they can't use up the guests' cap.
"""

import logging
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Any

import httpx
from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth import CurrentUser, current_user
from app.config import get_settings
from app.db import DailyUsage, SessionLocal, UserDailyUsage
from app.finder.github import GITHUB_API_URL

logger = logging.getLogger(__name__)

RATE_WINDOW_SECONDS = 3600.0
MAX_TRACKED_CLIENTS = 10_000
GITHUB_TIMEOUT_SECONDS = 15.0
KB_PER_MB = 1024
INDEX_PASSWORD_MESSAGE = "This demo only indexes the repos it comes with. Adding or changing repos needs the index password."
INDEXING_OFF_MESSAGE = "This demo only indexes the repos it comes with, so adding or changing repos is turned off."
# What each limited action is called in messages, and the setting that holds its hourly limit.
ACTIONS = {
    "search": ("searches", "rate_limit_search_per_hour"),
    "ask": ("questions", "rate_limit_ask_per_hour"),
    "guide": ("guides", "rate_limit_guide_per_hour"),
    "index": ("attempts to change repos", "rate_limit_index_per_hour"),
}


class RateLimiter:
    """Counts each client's recent requests per action, in memory.

    A restart forgets the counts, which is fine: the daily cap, kept in the database, is the hard limit.
    """

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def wait_seconds(self, action: str, client: str, limit: int) -> float | None:
        """Record a request and return None, or return how long to wait if the client is over its limit."""
        now = time.monotonic()
        with self._lock:
            # Forgetting everyone is rare and harmless, and stops the counts growing without end.
            if len(self._hits) > MAX_TRACKED_CLIENTS:
                self._hits.clear()
            hits = self._hits.setdefault((action, client), deque())
            while hits and hits[0] <= now - RATE_WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= limit:
                return hits[0] + RATE_WINDOW_SECONDS - now
            hits.append(now)
            return None

    def reset(self) -> None:
        """Forget every client."""
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()


def client_ip(request: Request) -> str:
    """Return who is asking. Behind a proxy, every request appears to come from the proxy itself.

    CLIENT_IP_HEADER names the header the proxy fills in. Other headers can be set by the client,
    so they aren't trusted. docs/DEPLOY.md has a check that this one can't be faked.
    """
    header = get_settings().client_ip_header
    value = request.headers.get(header, "") if header else ""
    if value.strip():
        return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate(action: str, request: Request, user: CurrentUser | None = None) -> None:
    """Raise a 429 if this user or guest has used up their hourly limit for this action."""
    label, setting = ACTIONS[action]
    limit = getattr(get_settings(), setting)
    client = f"user:{user.id}" if user else client_ip(request)
    wait = rate_limiter.wait_seconds(action, client, limit)
    if wait is not None:
        minutes = max(1, round(wait / 60))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"You've reached this demo's limit of {limit} {label} an hour. Please try again in about {minutes} min.",
            headers={"Retry-After": str(int(wait) + 1)},
        )


def _add_one(db: Session, model: type[DailyUsage] | type[UserDailyUsage], key: dict[str, Any], column: str) -> int:
    """Add one to a usage counter and return its new value.

    The upsert is one atomic statement, so simultaneous requests can't lose a count.
    """
    counter = getattr(model, column)
    statement = (
        insert(model)
        .values(**key, **{column: 1})
        .on_conflict_do_update(index_elements=list(key), set_={column: counter + 1})
        .returning(counter)
    )
    return db.scalar(statement)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _database_unavailable(error: SQLAlchemyError) -> HTTPException:
    """Without the count there's no way to protect the API quota, so the request has to be refused."""
    logger.warning("Couldn't count a demo request: %s", type(error).__name__)
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The demo's database is unavailable. Try again shortly.")


def _count_daily_request() -> None:
    """Count a guest's request towards today's total, and raise a friendly 429 once that's over the cap.

    The count lives in the database, so a restart or a free-tier spin-down can't reset it.
    """
    cap = get_settings().daily_request_cap
    try:
        with SessionLocal() as db:
            used = _add_one(db, DailyUsage, {"day": _today()}, "requests")
            db.commit()
    except SQLAlchemyError as error:
        raise _database_unavailable(error) from error
    if used > cap:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Demo limit reached: this public demo handles {cap} requests a day, and today's are used up. "
            "It resets at midnight UTC. You can also run RepoScout yourself from its GitHub repo.",
        )


def _count_user_request(user: CurrentUser) -> None:
    """Count a signed-in user's request, against their own cap and then the total for all signed-in users."""
    settings = get_settings()
    today = _today()
    message = None
    try:
        with SessionLocal() as db:
            if _add_one(db, UserDailyUsage, {"user_id": user.id, "day": today}, "requests") > settings.user_daily_request_cap:
                message = f"You've used your {settings.user_daily_request_cap} requests for today. They reset at midnight UTC."
            elif _add_one(db, DailyUsage, {"day": today}, "user_requests") > settings.signed_in_daily_request_cap:
                message = "Demo limit reached: signed-in users have used today's requests. They reset at midnight UTC."
            db.commit()
    except SQLAlchemyError as error:
        raise _database_unavailable(error) from error
    if message:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, message)


def limit(action: str) -> Callable[..., None]:
    """Return a FastAPI dependency that applies the demo's limits to an expensive endpoint."""

    def check(request: Request, user: CurrentUser | None = Depends(current_user)) -> None:
        if not get_settings().demo_mode:
            return
        _check_rate(action, request, user)
        if user:
            _count_user_request(user)
        else:
            _count_daily_request()

    return check


def check_index_rate(request: Request, user: CurrentUser) -> None:
    """In demo mode, apply the hourly limit on a signed-in user's attempts to index repos."""
    if get_settings().demo_mode:
        _check_rate("index", request, user)


def count_user_index(user: CurrentUser) -> None:
    """In demo mode, count an indexing job against the user's repos per day, or raise a 429 over that limit."""
    settings = get_settings()
    if not settings.demo_mode:
        return
    try:
        with SessionLocal() as db:
            used = _add_one(db, UserDailyUsage, {"user_id": user.id, "day": _today()}, "indexes")
            db.commit()
    except SQLAlchemyError as error:
        raise _database_unavailable(error) from error
    if used > settings.user_repos_per_day:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"You've reached your limit of {settings.user_repos_per_day} indexed repos a day. It resets at midnight UTC.",
        )


def reserve_signed_in_chunks(count: int) -> bool:
    """Take chunks from signed-in users' shared daily budget, or return False if too few are left.

    Every chunk uses one of Gemini's daily embeddings, which guests' questions need too. The condition
    is part of the upsert, so simultaneous jobs can't overspend the budget together.
    """
    budget = get_settings().signed_in_chunks_per_day
    if count > budget:
        return False
    statement = (
        insert(DailyUsage)
        .values(day=_today(), user_chunks=count)
        .on_conflict_do_update(
            index_elements=[DailyUsage.day],
            set_={"user_chunks": DailyUsage.user_chunks + count},
            where=DailyUsage.user_chunks + count <= budget,
        )
        .returning(DailyUsage.user_chunks)
    )
    with SessionLocal() as db:
        used = db.scalar(statement)
        db.commit()
    return used is not None


def require_index_password(request: Request, x_index_password: str = Header(default="")) -> None:
    """In demo mode, only someone with INDEX_PASSWORD may add, re-index or delete repos.

    Attempts are rate limited, and compare_digest takes the same time wherever a guess goes wrong,
    so the password can't be found by trying or by timing.
    """
    settings = get_settings()
    if not settings.demo_mode:
        return
    expected = settings.index_password.get_secret_value()
    if not expected:
        raise HTTPException(status.HTTP_403_FORBIDDEN, INDEXING_OFF_MESSAGE)
    _check_rate("index", request)
    if not secrets.compare_digest(x_index_password.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INDEX_PASSWORD_MESSAGE)


def _repo_size_mb(name: str) -> float:
    """Ask GitHub how big a repo is, without cloning it."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "RepoScout"}
    token = get_settings().github_token.get_secret_value()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.get(f"{GITHUB_API_URL}/repos/{name}", headers=headers, timeout=GITHUB_TIMEOUT_SECONDS, follow_redirects=True)
    except httpx.HTTPError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Couldn't check the repo's size on GitHub.") from error
    if response.status_code == status.HTTP_404_NOT_FOUND:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"GitHub has no public repo called {name}.")
    if response.status_code != status.HTTP_200_OK:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Couldn't check the repo's size on GitHub.")
    return response.json().get("size", 0) / KB_PER_MB


def check_repo_size(name: str) -> None:
    """In demo mode, refuse repos over MAX_REPO_SIZE_MB, before any time goes into cloning them."""
    settings = get_settings()
    if not settings.demo_mode:
        return
    size = _repo_size_mb(name)
    if size > settings.max_repo_size_mb:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"{name} is about {size:.0f} MB, and this demo's limit is {settings.max_repo_size_mb} MB.",
        )
