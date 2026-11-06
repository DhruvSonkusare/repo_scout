"""Sign in with GitHub, using OAuth's authorization code flow."""

import logging
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ValidationError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from app.auth import SESSION_COOKIE, SESSION_MAX_AGE_SECONDS, CurrentUser, secure_cookies, session_token, session_user
from app.config import get_settings
from app.db import SessionLocal, User
from app.finder.github import GITHUB_API_URL
from app.schemas import MeOut, UserOut

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_TIMEOUT_SECONDS = 15.0
STATE_COOKIE = "reposcout_oauth_state"
STATE_COOKIE_PATH = "/auth/github"
STATE_MAX_AGE_SECONDS = 600
LOGIN_FAILED_REDIRECT = "/?notice=login_failed"


class GitHubProfile(BaseModel):
    """The parts of a GitHub profile we keep."""

    id: int
    login: str
    avatar_url: str | None = None


class LoginError(Exception):
    """GitHub didn't complete the sign-in."""


def _http_client() -> httpx.Client:
    """The client for GitHub's login endpoints. Tests replace this with a mock."""
    return httpx.Client(timeout=GITHUB_TIMEOUT_SECONDS)


def _redirect_uri() -> str:
    return f"{get_settings().public_base_url}/auth/github/callback"


def _require_login_enabled() -> None:
    if not get_settings().login_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sign-in isn't set up on this server.")


def _fetch_profile(code: str) -> GitHubProfile:
    """Swap the one-time code for an access token, and use it to read the profile.

    The token stays in this function: it isn't saved or logged, and we ask for no scopes, so it
    could only read public profile data anyway.
    """
    settings = get_settings()
    try:
        with _http_client() as client:
            token_response = client.post(
                GITHUB_TOKEN_URL,
                headers={"Accept": "application/json"},
                data={
                    "client_id": settings.github_client_id,
                    "client_secret": settings.github_client_secret.get_secret_value(),
                    "code": code,
                    "redirect_uri": _redirect_uri(),
                },
            )
            # GitHub reports a bad code with a 200 and an error in the body, so look for the token itself.
            access_token = token_response.json().get("access_token") if token_response.status_code == status.HTTP_200_OK else None
            if not access_token:
                raise LoginError("GitHub didn't issue a token.")
            profile_response = client.get(
                f"{GITHUB_API_URL}/user",
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {access_token}",
                    "User-Agent": "RepoScout",
                },
            )
            if profile_response.status_code != status.HTTP_200_OK:
                raise LoginError(f"GitHub's profile request returned {profile_response.status_code}.")
            return GitHubProfile.model_validate(profile_response.json())
    except (httpx.HTTPError, ValueError, AttributeError, ValidationError) as error:
        raise LoginError(f"GitHub's reply couldn't be used ({type(error).__name__}).") from error


def _save_user(profile: GitHubProfile) -> int:
    """Create the user, or refresh a returning user's name and picture, and return their id."""
    values = {"login": profile.login, "avatar_url": profile.avatar_url}
    statement = (
        insert(User)
        .values(github_id=profile.id, **values)
        .on_conflict_do_update(index_elements=[User.github_id], set_=values)
        .returning(User.id)
    )
    with SessionLocal() as db:
        user_id = db.scalars(statement).one()
        db.commit()
    return user_id


def _clear_state_cookie(response: Response) -> None:
    response.delete_cookie(STATE_COOKIE, path=STATE_COOKIE_PATH, secure=secure_cookies(), httponly=True, samesite="lax")


@router.get("/auth/github/login", include_in_schema=False, dependencies=[Depends(_require_login_enabled)])
def login() -> RedirectResponse:
    """Send the visitor to GitHub, remembering a random state in a cookie to check when they come back.

    Without that check, an attacker could send someone a callback link holding the attacker's own
    code, signing the victim in to the attacker's account, where their private work would be saved.
    """
    state = secrets.token_urlsafe(32)
    query = urlencode({"client_id": get_settings().github_client_id, "redirect_uri": _redirect_uri(), "state": state})
    response = RedirectResponse(f"{GITHUB_AUTHORIZE_URL}?{query}", status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        STATE_COOKIE,
        state,
        max_age=STATE_MAX_AGE_SECONDS,
        path=STATE_COOKIE_PATH,
        secure=secure_cookies(),
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/auth/github/callback", include_in_schema=False, dependencies=[Depends(_require_login_enabled)])
def callback(request: Request, code: str = "", state: str = "") -> RedirectResponse:
    """Finish signing in. Every failure, including refusing on GitHub, returns to the page as a guest."""
    expected = request.cookies.get(STATE_COOKIE, "")
    failed = RedirectResponse(LOGIN_FAILED_REDIRECT, status_code=status.HTTP_303_SEE_OTHER)
    _clear_state_cookie(failed)
    if not expected or not secrets.compare_digest(state.encode(), expected.encode()):
        logger.warning("Refused a GitHub sign-in: the state didn't match.")
        return failed
    if not code:
        return failed
    try:
        user_id = _save_user(_fetch_profile(code))
    except LoginError as error:
        logger.warning("GitHub sign-in failed: %s", error)
        return failed
    except SQLAlchemyError as error:
        logger.warning("Couldn't save a user who signed in: %s", type(error).__name__)
        return failed

    response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    _clear_state_cookie(response)
    # HttpOnly keeps it from scripts, and SameSite=Lax keeps it off other sites' POST requests.
    response.set_cookie(
        SESSION_COOKIE,
        session_token(user_id),
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
        secure=secure_cookies(),
        httponly=True,
        samesite="lax",
    )
    return response


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT, include_in_schema=False)
def logout(response: Response) -> None:
    """Sign out. It's a POST so that another site can't sign someone out with a link or an image."""
    response.delete_cookie(SESSION_COOKIE, path="/", secure=secure_cookies(), httponly=True, samesite="lax")


@router.get("/api/me", response_model=MeOut)
def me(user: CurrentUser | None = Depends(session_user)) -> MeOut:
    """Tell the frontend who is signed in, and whether signing in is possible here."""
    return MeOut(
        login_enabled=get_settings().login_enabled,
        user=UserOut(id=user.id, login=user.login, avatar_url=user.avatar_url) if user else None,
    )
