"""Tests for the GitHub client, using httpx.MockTransport so nothing touches the network."""

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.config import get_settings
from app.finder.github import MAX_README_CHARS, GitHubClient, GitHubError, GitHubRateLimitError

pytestmark = pytest.mark.usefixtures("fresh_settings")

Handler = Callable[[httpx.Request], httpx.Response]

FULL_ITEM = {
    "full_name": "acme/tables",
    "html_url": "https://github.com/acme/tables",
    "description": "Extract tables",
    "stargazers_count": 1200,
    "language": "Python",
    "topics": ["pdf", "tables"],
    "pushed_at": "2026-03-01T10:30:00Z",
    "license": {"key": "mit", "name": "MIT License", "spdx_id": "MIT"},
    "owner": {"login": "acme"},
}
SPARSE_ITEM = {
    "full_name": "acme/bare",
    "html_url": "https://github.com/acme/bare",
    "description": None,
    "stargazers_count": 0,
    "language": None,
    "topics": [],
    "pushed_at": None,
    "license": None,
}


@pytest.fixture(autouse=True)
def no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start without a token, whatever the developer's .env says."""
    monkeypatch.setenv("GITHUB_TOKEN", "")


def search(handler: Handler, *args: Any, **kwargs: Any) -> Any:
    """Run search_repos against a fake GitHub."""

    async def run() -> Any:
        async with GitHubClient(transport=httpx.MockTransport(handler)) as github:
            return await github.search_repos(*args, **kwargs)

    return asyncio.run(run())


def readme(handler: Handler, full_name: str = "acme/tables") -> str:
    """Run get_readme against a fake GitHub."""

    async def run() -> str:
        async with GitHubClient(transport=httpx.MockTransport(handler)) as github:
            return await github.get_readme(full_name)

    return asyncio.run(run())


def respond_with_items(*items: dict[str, Any]) -> Handler:
    return lambda request: httpx.Response(200, json={"total_count": len(items), "items": list(items)})


def recording(requests: list[httpx.Request], response: httpx.Response) -> Handler:
    """Answer every request with this response, recording the requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return handler


# --- search_repos


def test_search_converts_results_to_dataclasses() -> None:
    repos = search(respond_with_items(FULL_ITEM), "pdf tables")

    repo = repos[0]
    assert repo.full_name == "acme/tables"
    assert repo.html_url == "https://github.com/acme/tables"
    assert repo.description == "Extract tables"
    assert repo.stars == 1200
    assert repo.language == "Python"
    assert repo.topics == ["pdf", "tables"]
    assert repo.pushed_at == datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc)
    assert repo.license == "MIT"


def test_search_handles_results_with_missing_values() -> None:
    repo = search(respond_with_items(SPARSE_ITEM), "bare")[0]

    assert repo.description is None
    assert repo.language is None
    assert repo.topics == []
    assert repo.pushed_at is None
    assert repo.license is None


def test_a_license_github_cannot_identify_is_called_other() -> None:
    item = {**SPARSE_ITEM, "license": {"key": "other", "name": "Other", "spdx_id": "NOASSERTION"}}

    assert search(respond_with_items(item), "bare")[0].license == "Other"


def test_search_with_no_results_returns_an_empty_list() -> None:
    assert search(respond_with_items(), "nothing matches") == []


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (None, "pdf tables archived:false"),
        ("Python", "pdf tables language:Python archived:false"),
        ("Jupyter Notebook", 'pdf tables language:"Jupyter Notebook" archived:false'),
    ],
)
def test_search_sends_the_query_sorted_by_stars(language: str | None, expected: str) -> None:
    requests: list[httpx.Request] = []

    search(recording(requests, httpx.Response(200, json={"items": []})), "pdf tables", language=language, per_page=5)

    url = requests[0].url
    assert url.host == "api.github.com"
    assert url.path == "/search/repositories"
    assert url.params["q"] == expected
    assert url.params["sort"] == "stars"
    assert url.params["order"] == "desc"
    assert url.params["per_page"] == "5"


def test_search_asks_for_ten_results_by_default() -> None:
    requests: list[httpx.Request] = []

    search(recording(requests, httpx.Response(200, json={"items": []})), "pdf")

    assert requests[0].url.params["per_page"] == "10"


# --- headers


def test_requests_send_githubs_headers() -> None:
    requests: list[httpx.Request] = []

    search(recording(requests, httpx.Response(200, json={"items": []})), "pdf")

    headers = requests[0].headers
    assert headers["accept"] == "application/vnd.github+json"
    assert headers["x-github-api-version"] == "2022-11-28"
    assert headers["user-agent"] == "RepoGuide"


def test_no_token_sends_no_authorization_header() -> None:
    requests: list[httpx.Request] = []

    search(recording(requests, httpx.Response(200, json={"items": []})), "pdf")

    assert "authorization" not in requests[0].headers


def test_a_token_is_sent_as_a_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    requests: list[httpx.Request] = []

    search(recording(requests, httpx.Response(200, json={"items": []})), "pdf")

    assert requests[0].headers["authorization"] == "Bearer test-token"


# --- get_readme


def test_readme_asks_for_raw_content() -> None:
    requests: list[httpx.Request] = []

    assert readme(recording(requests, httpx.Response(200, text="# Tables"))) == "# Tables"

    assert requests[0].url.path == "/repos/acme/tables/readme"
    assert requests[0].headers["accept"] == "application/vnd.github.raw"


def test_readme_decodes_utf8_when_github_names_no_charset() -> None:
    response = httpx.Response(200, content="naïve — 表".encode(), headers={"content-type": "application/vnd.github.raw"})

    assert readme(lambda request: response) == "naïve — 表"


def test_readme_is_cut_to_the_limit() -> None:
    long_readme = "x" * (MAX_README_CHARS + 500)

    assert len(readme(lambda request: httpx.Response(200, text=long_readme))) == MAX_README_CHARS


def test_a_repo_without_a_readme_returns_an_empty_string() -> None:
    assert readme(lambda request: httpx.Response(404, json={"message": "Not Found"})) == ""


def test_a_renamed_repo_follows_the_redirect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/acme/old/readme":
            return httpx.Response(301, headers={"location": "https://api.github.com/repositories/1/readme"})
        return httpx.Response(200, text="# Moved")

    assert readme(handler, "acme/old") == "# Moved"


# --- rate limits


RATE_LIMITED_RESPONSES = {
    "429": lambda: httpx.Response(429, json={"message": "Too many requests"}),
    "403 with no requests left": lambda: httpx.Response(
        403, headers={"x-ratelimit-remaining": "0"}, json={"message": "Forbidden"}
    ),
    "403 with retry-after": lambda: httpx.Response(403, headers={"retry-after": "30"}, json={"message": "Forbidden"}),
    "403 secondary limit": lambda: httpx.Response(
        403, json={"message": "You have exceeded a secondary rate limit."}
    ),
}


@pytest.mark.parametrize("build", RATE_LIMITED_RESPONSES.values(), ids=RATE_LIMITED_RESPONSES.keys())
def test_rate_limits_raise_a_clear_error_from_both_calls(build: Callable[[], httpx.Response]) -> None:
    with pytest.raises(GitHubRateLimitError, match="rate limit was reached"):
        search(lambda request: build(), "pdf")
    with pytest.raises(GitHubRateLimitError, match="rate limit was reached"):
        readme(lambda request: build())


def test_rate_limit_message_uses_retry_after() -> None:
    response = httpx.Response(429, headers={"retry-after": "90"})

    with pytest.raises(GitHubRateLimitError, match="about 2 minute"):
        search(lambda request: response, "pdf")


def test_rate_limit_message_uses_the_reset_time() -> None:
    reset_at = int(time.time()) + 600
    headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset_at)}

    with pytest.raises(GitHubRateLimitError, match=r"about (10|11) minute"):
        search(lambda request: httpx.Response(403, headers=headers), "pdf")


def test_rate_limit_message_without_a_wait_time() -> None:
    with pytest.raises(GitHubRateLimitError, match="Try again shortly"):
        search(lambda request: httpx.Response(429), "pdf")


def test_rate_limit_message_suggests_a_token_only_if_there_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(GitHubRateLimitError, match="GITHUB_TOKEN"):
        search(lambda request: httpx.Response(429), "pdf")

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    get_settings.cache_clear()
    with pytest.raises(GitHubRateLimitError) as error:
        search(lambda request: httpx.Response(429), "pdf")
    assert "GITHUB_TOKEN" not in str(error.value)


# --- other errors


def test_a_403_that_is_not_a_rate_limit_is_not_called_one() -> None:
    with pytest.raises(GitHubError, match="HTTP 403") as error:
        search(lambda request: httpx.Response(403, json={"message": "Repository access blocked"}), "pdf")

    assert not isinstance(error.value, GitHubRateLimitError)


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "GITHUB_TOKEN"),
        (422, "search query"),
        (500, "HTTP 500"),
        (503, "HTTP 503"),
    ],
)
def test_other_errors_have_clear_messages(status: int, message: str) -> None:
    with pytest.raises(GitHubError, match=message):
        search(lambda request: httpx.Response(status, json={"message": "test"}), "pdf")


def test_a_readme_server_error_is_not_mistaken_for_a_missing_readme() -> None:
    with pytest.raises(GitHubError, match="HTTP 500"):
        readme(lambda request: httpx.Response(500))


@pytest.mark.parametrize("error", [httpx.ConnectError("no route"), httpx.ReadTimeout("slow")])
def test_network_failures_raise_a_clear_error(error: httpx.HTTPError) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(GitHubError, match="Couldn't reach GitHub"):
        search(handler, "pdf")
    with pytest.raises(GitHubError, match="Couldn't reach GitHub"):
        readme(handler)


def test_error_messages_never_contain_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-value")

    for status in [401, 403, 429, 500]:
        with pytest.raises(GitHubError) as error:
            search(lambda request: httpx.Response(status), "pdf")
        assert "test-token-value" not in str(error.value)
