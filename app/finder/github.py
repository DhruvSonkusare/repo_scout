"""Search GitHub for repos and read their READMEs through the REST API."""

import time
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any, Self

import httpx

from app.config import get_settings

GITHUB_API_URL = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_README_CHARS = 4000
RAW_CONTENT_TYPE = "application/vnd.github.raw"


class GitHubError(Exception):
    """A GitHub request failed. The message is safe to show to users."""


class GitHubRateLimitError(GitHubError):
    """GitHub's rate limit was reached."""


@dataclass(frozen=True)
class RepoCandidate:
    """One repo from a GitHub search."""

    full_name: str
    html_url: str
    description: str | None
    stars: int
    language: str | None
    topics: list[str]
    pushed_at: datetime | None
    license: str | None


def _to_candidate(item: dict[str, Any]) -> RepoCandidate:
    """Convert one search result, keeping only the fields we use."""
    pushed_at = item.get("pushed_at")
    license_info = item.get("license") or {}
    # GitHub reports a license it can't identify as NOASSERTION, which means nothing to a reader.
    spdx_id = license_info.get("spdx_id")
    return RepoCandidate(
        full_name=item["full_name"],
        html_url=item["html_url"],
        description=item.get("description"),
        stars=item.get("stargazers_count", 0),
        language=item.get("language"),
        topics=item.get("topics") or [],
        pushed_at=datetime.fromisoformat(pushed_at) if pushed_at else None,
        license="Other" if spdx_id == "NOASSERTION" else spdx_id,
    )


def _build_search_query(query: str, language: str | None) -> str:
    """Add GitHub's search qualifiers. Language names with spaces, like Jupyter Notebook, need quotes."""
    parts = [query]
    if language:
        parts.append(f'language:"{language}"' if " " in language else f"language:{language}")
    parts.append("archived:false")
    return " ".join(parts)


def _is_rate_limited(response: httpx.Response) -> bool:
    """Return True for a rate limit. GitHub also uses 403 for other refusals, so the status isn't enough."""
    if response.status_code == 429:
        return True
    if response.status_code != 403:
        return False
    has_limit_header = response.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in response.headers
    return has_limit_header or "rate limit" in response.text.lower()


def _seconds_until_retry(response: httpx.Response) -> int | None:
    """Return how long GitHub says to wait, if it says."""
    retry_after = response.headers.get("retry-after", "")
    if retry_after.isdigit():
        return int(retry_after)
    reset_at = response.headers.get("x-ratelimit-reset", "")
    if reset_at.isdigit():
        return max(int(reset_at) - int(time.time()), 0)
    return None


class GitHubClient:
    """An async GitHub client. Use it as `async with GitHubClient() as github:` to close it afterwards."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._token = get_settings().github_token.get_secret_value()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "RepoScout",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API_URL,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            # GitHub redirects renamed repos. httpx drops the token if a redirect leaves this host.
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self._client.aclose()

    async def search_repos(self, query: str, language: str | None = None, per_page: int = 10) -> list[RepoCandidate]:
        """Return non-archived repos matching a query, with the most stars first."""
        params = {"q": _build_search_query(query, language), "sort": "stars", "order": "desc", "per_page": per_page}
        response = await self._get("/search/repositories", params=params)
        self._raise_for_status(response)
        return [_to_candidate(item) for item in response.json().get("items", [])]

    async def get_readme(self, full_name: str) -> str:
        """Return the start of a repo's README, or an empty string if it has none."""
        response = await self._get(f"/repos/{full_name}/readme", headers={"Accept": RAW_CONTENT_TYPE})
        if response.status_code == 404:
            return ""
        self._raise_for_status(response)
        return response.text[:MAX_README_CHARS]

    async def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        """Send a GET request, turning network failures into a GitHubError."""
        try:
            return await self._client.get(path, **kwargs)
        except httpx.HTTPError as error:
            raise GitHubError("Couldn't reach GitHub. Check your internet connection.") from error

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise a clear error for anything but success."""
        if response.is_success:
            return
        if _is_rate_limited(response):
            raise GitHubRateLimitError(self._rate_limit_message(response))
        if response.status_code == 401:
            raise GitHubError("GitHub rejected the token. Check GITHUB_TOKEN in .env.")
        if response.status_code == 422:
            raise GitHubError("GitHub couldn't understand that search query.")
        raise GitHubError(f"GitHub returned an error (HTTP {response.status_code}).")

    def _rate_limit_message(self, response: httpx.Response) -> str:
        """Say how long to wait, and suggest a token if there isn't one."""
        seconds = _seconds_until_retry(response)
        wait = "Try again shortly." if seconds is None else f"Try again in about {seconds // 60 + 1} minute(s)."
        advice = "" if self._token else " Setting GITHUB_TOKEN in .env allows many more requests."
        return f"GitHub's rate limit was reached. {wait}{advice}"
