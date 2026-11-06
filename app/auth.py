"""Who is asking, and what they may see: the signed session cookie and the access rules.

A guest sees public rows. A signed-in user sees their own rows, the public repos and the repos they added.
Someone else's private item answers 404, so its existence isn't revealed.
"""

from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status
from itsdangerous import BadSignature, URLSafeTimedSerializer
from sqlalchemy import ColumnElement, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import InstrumentedAttribute, Session

from app.config import get_settings
from app.db import Repo, SessionLocal, User, UserRepo

SESSION_COOKIE = "reposcout_session"
SESSION_MAX_AGE_SECONDS = 14 * 24 * 3600
SESSION_SALT = "reposcout-session"
SESSION_ENDED_MESSAGE = "Your session has ended, so nothing was saved. Sign in again to use your workspace."


@dataclass(frozen=True)
class CurrentUser:
    """The signed-in user. A plain copy, so a long request doesn't hold a database connection for it."""

    id: int
    login: str
    avatar_url: str | None


def _serializer() -> URLSafeTimedSerializer:
    """Signs cookies, so a visitor can read theirs but can't change the user id in it."""
    return URLSafeTimedSerializer(get_settings().session_secret.get_secret_value(), salt=SESSION_SALT)


def session_token(user_id: int) -> str:
    """Return the cookie value that signs this user in."""
    return _serializer().dumps({"user_id": user_id})


def secure_cookies() -> bool:
    """Whether cookies get the Secure flag. Browsers refuse those over plain HTTP, as on localhost."""
    return get_settings().public_base_url.startswith("https://")


def session_user(request: Request) -> CurrentUser | None:
    """Return the user this request's cookie signs in. Anything invalid, expired or unknown counts as a guest."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token or not get_settings().login_enabled:
        return None
    try:
        user_id = _serializer().loads(token, max_age=SESSION_MAX_AGE_SECONDS)["user_id"]
    except (BadSignature, KeyError, TypeError):
        return None
    try:
        with SessionLocal() as db:
            user = db.get(User, user_id)
            return CurrentUser(user.id, user.login, user.avatar_url) if user else None
    except SQLAlchemyError as error:
        # Treating this as a guest could publish what a signed-in user is about to send.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The database is unavailable. Try again shortly.") from error


def current_user(request: Request, x_signed_in_as: str = Header(default="")) -> CurrentUser | None:
    """FastAPI dependency for the user, or None for a guest.

    A page that was signed in says so in this header. If that session has since ended, or another tab
    changed account, the request is refused. Otherwise it would run as someone else, and a private
    question could end up in the public space.
    """
    user = session_user(request)
    if x_signed_in_as and (user is None or str(user.id) != x_signed_in_as):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, SESSION_ENDED_MESSAGE, headers={"X-Session-Ended": "true"})
    return user


def owner_id(user: CurrentUser | None) -> int | None:
    """The owner of a row in this user's space. None is the public space."""
    return user.id if user else None


def owned_by(column: InstrumentedAttribute[int | None], user: CurrentUser | None) -> ColumnElement[bool]:
    """A query filter for the rows in this user's space."""
    return column == user.id if user else column.is_(None)


def visible_repos(user: CurrentUser | None) -> ColumnElement[bool]:
    """A query filter for the repos this user can see."""
    if user is None:
        return Repo.is_public
    return or_(Repo.is_public, Repo.id.in_(select(UserRepo.repo_id).where(UserRepo.user_id == user.id)))


def in_workspace(db: Session, repo_id: int, user: CurrentUser | None) -> bool:
    """Whether this user added this repo."""
    return user is not None and db.get(UserRepo, (user.id, repo_id)) is not None


def can_see_repo(db: Session, repo: Repo, user: CurrentUser | None) -> bool:
    """Whether this user can see this repo."""
    return repo.is_public or in_workspace(db, repo.id, user)


def get_repo_or_404(db: Session, repo_id: int, user: CurrentUser | None) -> Repo:
    """Return a repo this user can see, or raise a 404."""
    repo = db.get(Repo, repo_id)
    if repo is None or not can_see_repo(db, repo, user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not found.")
    return repo

