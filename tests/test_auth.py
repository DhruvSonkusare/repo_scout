"""Tests for Sign in with GitHub and for who can see what. GitHub is mocked, and no real calls are made."""

import json
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import Conversation, DailyUsage, Repo, Search, SearchResult, User, UserDailyUsage, UserRepo
from app.auth import SESSION_COOKIE
from app.main import app
from app.finder.chains import Candidate, RepoResult, SearchPlan
from app.finder.github import RepoCandidate
from app.routers.auth import LOGIN_FAILED_REDIRECT, STATE_COOKIE

pytestmark = pytest.mark.usefixtures("fresh_settings", "cloned_urls")

ACCESS_TOKEN = "gho_the-access-token"
ALICE = {"id": 1001, "login": "alice", "avatar_url": "https://avatars.example/alice"}
BOB = {"id": 1002, "login": "bob", "avatar_url": "https://avatars.example/bob"}
LOGIN_SETTINGS = {
    "GITHUB_CLIENT_ID": "the-client-id",
    "GITHUB_CLIENT_SECRET": "the-client-secret",
    "SESSION_SECRET": "a-test-session-secret-of-a-good-length",
    "PUBLIC_BASE_URL": "http://localhost:8000/",
}
PASSWORD = {"X-Index-Password": "the-index-password"}
REPO_URL = "https://github.com/acme/sample"
OTHER_URL = "https://github.com/acme/other"
BASE_URL = "http://localhost:8000"


class FakeGitHub:
    """GitHub's login endpoints. `profile` is whoever approves the next sign-in."""

    def __init__(self) -> None:
        self.profile: dict[str, Any] = ALICE
        self.token_reply: dict[str, Any] = {"access_token": ACCESS_TOKEN, "token_type": "bearer"}
        self.profile_status = 200
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json=self.token_reply)
        assert request.headers["Authorization"] == f"Bearer {ACCESS_TOKEN}"
        return httpx.Response(self.profile_status, json=self.profile)


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> FakeGitHub:
    """Turn login on, with a fake GitHub behind it."""
    for name, value in LOGIN_SETTINGS.items():
        monkeypatch.setenv(name, value)
    fake = FakeGitHub()
    monkeypatch.setattr("app.routers.auth._http_client", lambda: httpx.Client(transport=httpx.MockTransport(fake)))
    return fake


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch, github: FakeGitHub) -> pytest.MonkeyPatch:
    """Turn demo mode on as well, with small repos and no calls to GitHub's API."""
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("INDEX_PASSWORD", PASSWORD["X-Index-Password"])
    monkeypatch.setattr("app.demo._repo_size_mb", lambda name: 1.0)
    return monkeypatch


def visitor() -> TestClient:
    """A browser of its own, with its own cookies. Redirects aren't followed, so tests can inspect them."""
    return TestClient(app, base_url=BASE_URL, follow_redirects=False)


def sign_in(github: FakeGitHub, profile: dict[str, Any] = ALICE) -> TestClient:
    """Go through the whole login flow in a new browser."""
    github.profile = profile
    browser = visitor()
    state = parse_qs(urlparse(browser.get("/auth/github/login").headers["location"]).query)["state"][0]
    response = browser.get("/auth/github/callback", params={"code": "the-code", "state": state})
    assert response.headers["location"] == "/"
    return browser


@pytest.fixture
def alice(github: FakeGitHub) -> TestClient:
    return sign_in(github, ALICE)


@pytest.fixture
def bob(github: FakeGitHub) -> TestClient:
    return sign_in(github, BOB)


@pytest.fixture
def guest(github: FakeGitHub) -> TestClient:
    return visitor()


def add_repo(client: TestClient, url: str = REPO_URL, headers: dict[str, str] | None = None) -> dict[str, Any]:
    response = client.post("/api/repos", json={"url": url}, headers=headers or {})
    assert response.status_code in (200, 201), response.text
    return response.json()


def ask(client: TestClient, repo_id: int, question: str = "Hi?") -> int:
    response = client.post(f"/api/repos/{repo_id}/ask", json={"question": question})
    assert response.status_code == 200, response.text
    return json.loads(response.text.splitlines()[0])["conversation_id"]


def save_search(db: Session, problem: str, owner: int | None) -> tuple[int, int]:
    """Save a search with one result, and return both ids."""
    search = Search(problem=problem, plan={"queries": ["q"]}, owner_user_id=owner)
    search.results = [
        SearchResult(
            owner_user_id=owner,
            full_name="acme/sample",
            url=REPO_URL,
            stars=1,
            score=90,
            why_it_fits="It fits.",
            caveats="None.",
            how_to_start=[],
            problem_guide="A private guide." if owner else "A public guide.",
        )
    ]
    db.add(search)
    db.commit()
    return search.id, search.results[0].id


def user_id(db: Session, login: str) -> int:
    return db.scalars(select(User.id).where(User.login == login)).one()


# --- the login flow


def test_login_redirects_to_github_with_a_random_state(github: FakeGitHub) -> None:
    first, second = visitor().get("/auth/github/login"), visitor().get("/auth/github/login")

    location = urlparse(first.headers["location"])
    query = parse_qs(location.query)
    assert first.status_code == 302
    assert f"{location.scheme}://{location.netloc}{location.path}" == "https://github.com/login/oauth/authorize"
    assert query["client_id"] == ["the-client-id"]
    assert query["redirect_uri"] == ["http://localhost:8000/auth/github/callback"]
    assert "scope" not in query
    assert len(query["state"][0]) >= 32
    assert query["state"] != parse_qs(urlparse(second.headers["location"]).query)["state"]


def test_the_login_flow_signs_the_user_in(github: FakeGitHub, db_session: Session) -> None:
    browser = sign_in(github)

    assert browser.get("/api/me").json() == {
        "login_enabled": True,
        "user": {"id": 1, "login": "alice", "avatar_url": ALICE["avatar_url"]},
    }
    user = db_session.scalars(select(User)).one()
    assert (user.github_id, user.login) == (1001, "alice")
    sent = parse_qs(github.requests[0].content.decode())
    assert sent["code"] == ["the-code"]
    assert sent["client_secret"] == ["the-client-secret"]
    assert sent["redirect_uri"] == ["http://localhost:8000/auth/github/callback"]


def test_the_state_cookie_is_cleared_after_login(github: FakeGitHub) -> None:
    browser = visitor()
    state = parse_qs(urlparse(browser.get("/auth/github/login").headers["location"]).query)["state"][0]

    browser.get("/auth/github/callback", params={"code": "the-code", "state": state})

    assert STATE_COOKIE not in browser.cookies


def test_a_returning_user_keeps_their_account_with_a_fresh_profile(github: FakeGitHub, db_session: Session) -> None:
    sign_in(github)

    sign_in(github, {**ALICE, "login": "alice-renamed"})

    assert [user.login for user in db_session.scalars(select(User))] == ["alice-renamed"]


def test_logout_ends_the_session(alice: TestClient) -> None:
    assert alice.post("/auth/logout").status_code == 204

    assert alice.get("/api/me").json()["user"] is None


def test_logout_cannot_be_triggered_by_a_link(alice: TestClient) -> None:
    assert alice.get("/auth/logout").status_code == 405


# --- refused sign-ins


@pytest.mark.parametrize("state", ["wrong-state", "", "ünïcode"])
def test_a_state_mismatch_is_rejected(github: FakeGitHub, db_session: Session, state: str) -> None:
    browser = visitor()
    browser.get("/auth/github/login")

    response = browser.get("/auth/github/callback", params={"code": "the-code", "state": state})

    assert response.headers["location"] == LOGIN_FAILED_REDIRECT
    assert SESSION_COOKIE not in response.cookies
    assert github.requests == []
    assert db_session.query(User).count() == 0
    assert browser.get("/api/me").json()["user"] is None


def test_a_callback_with_no_state_cookie_is_rejected(github: FakeGitHub) -> None:
    response = visitor().get("/auth/github/callback", params={"code": "the-code", "state": "any-state"})

    assert response.headers["location"] == LOGIN_FAILED_REDIRECT
    assert github.requests == []


def test_a_state_can_only_be_used_once(github: FakeGitHub) -> None:
    browser = visitor()
    state = parse_qs(urlparse(browser.get("/auth/github/login").headers["location"]).query)["state"][0]
    browser.get("/auth/github/callback", params={"code": "the-code", "state": state})
    browser.post("/auth/logout")

    response = browser.get("/auth/github/callback", params={"code": "the-code", "state": state})

    assert response.headers["location"] == LOGIN_FAILED_REDIRECT


def callback_with_valid_state(browser: TestClient, **params: str) -> httpx.Response:
    state = parse_qs(urlparse(browser.get("/auth/github/login").headers["location"]).query)["state"][0]
    return browser.get("/auth/github/callback", params={"state": state, **params})


def test_refusing_on_github_returns_as_a_guest(github: FakeGitHub) -> None:
    browser = visitor()

    response = callback_with_valid_state(browser, error="access_denied")

    assert response.headers["location"] == LOGIN_FAILED_REDIRECT
    assert github.requests == []


def test_a_bad_code_is_refused(github: FakeGitHub, db_session: Session) -> None:
    # GitHub answers a bad code with a 200 and an error in the body.
    github.token_reply = {"error": "bad_verification_code"}
    browser = visitor()

    response = callback_with_valid_state(browser, code="bad-code")

    assert response.headers["location"] == LOGIN_FAILED_REDIRECT
    assert db_session.query(User).count() == 0


@pytest.mark.parametrize("setup", ["status", "shape"])
def test_a_bad_profile_reply_is_refused(github: FakeGitHub, setup: str) -> None:
    if setup == "status":
        github.profile_status = 500
    else:
        github.profile = {"unexpected": True}
    browser = visitor()

    response = callback_with_valid_state(browser, code="the-code")

    assert response.headers["location"] == LOGIN_FAILED_REDIRECT


def test_github_being_unreachable_is_refused(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    monkeypatch.setattr("app.routers.auth._http_client", lambda: httpx.Client(transport=httpx.MockTransport(fail)))

    response = callback_with_valid_state(visitor(), code="the-code")

    assert response.headers["location"] == LOGIN_FAILED_REDIRECT


def test_a_database_failure_during_sign_in_returns_to_the_page(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(profile: Any) -> int:
        raise OperationalError("SELECT 1", {}, Exception("down"))

    monkeypatch.setattr("app.routers.auth._save_user", fail)

    assert callback_with_valid_state(visitor(), code="the-code").headers["location"] == LOGIN_FAILED_REDIRECT


# --- the session cookie


def set_cookie_header(response: httpx.Response, name: str) -> str:
    return next(value for value in response.headers.get_list("set-cookie") if value.startswith(f"{name}="))


def test_the_session_cookie_is_httponly_and_lax_but_not_secure_on_http(github: FakeGitHub) -> None:
    browser = visitor()
    state = parse_qs(urlparse(browser.get("/auth/github/login").headers["location"]).query)["state"][0]

    cookie = set_cookie_header(browser.get("/auth/github/callback", params={"code": "c", "state": state}), SESSION_COOKIE).lower()

    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "secure" not in cookie


def test_cookies_are_secure_in_production(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://reposcout.example")
    browser = TestClient(app, base_url="https://reposcout.example", follow_redirects=False)
    login = browser.get("/auth/github/login")
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]

    session = browser.get("/auth/github/callback", params={"code": "c", "state": state})

    for cookie in (set_cookie_header(login, STATE_COOKIE), set_cookie_header(session, SESSION_COOKIE)):
        assert "secure" in cookie.lower()
        assert "httponly" in cookie.lower()
    assert "https://reposcout.example/auth/github/callback" in parse_qs(urlparse(login.headers["location"]).query)["redirect_uri"]


def test_the_access_token_is_not_stored(github: FakeGitHub, db_session: Session) -> None:
    browser = visitor()
    state = parse_qs(urlparse(browser.get("/auth/github/login").headers["location"]).query)["state"][0]

    response = browser.get("/auth/github/callback", params={"code": "c", "state": state})

    assert ACCESS_TOKEN not in str(response.headers)
    row = db_session.execute(text("SELECT to_jsonb(users) FROM users")).scalar_one()
    assert ACCESS_TOKEN not in json.dumps(row)
    assert set(row) == {"id", "github_id", "login", "avatar_url", "created_at"}


@pytest.mark.parametrize("cookie", ["not-a-token", "eyJ1c2VyX2lkIjoxfQ.forged.signature"])
def test_a_forged_cookie_counts_as_a_guest(alice: TestClient, cookie: str) -> None:
    browser = visitor()
    browser.cookies.set(SESSION_COOKIE, cookie)

    assert browser.get("/api/me").json()["user"] is None


def test_a_cookie_signed_with_another_secret_counts_as_a_guest(alice: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SESSION_SECRET", "a-different-session-secret-of-good-length")
    get_settings.cache_clear()

    assert alice.get("/api/me").json()["user"] is None


def test_an_expired_session_counts_as_a_guest(alice: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.auth.SESSION_MAX_AGE_SECONDS", -1)

    assert alice.get("/api/me").json()["user"] is None


def test_a_session_for_a_deleted_user_counts_as_a_guest(alice: TestClient, db_session: Session) -> None:
    db_session.execute(text("DELETE FROM users"))
    db_session.commit()

    assert alice.get("/api/me").json()["user"] is None


def test_a_short_session_secret_stops_the_app_starting() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        Settings(session_secret="too-short")


# --- a page that thinks it's signed in can't fall into the public space


def test_a_page_whose_session_ended_cannot_post_publicly(alice: TestClient, db_session: Session) -> None:
    add_repo(alice)
    alice.post("/auth/logout")

    response = alice.post("/api/repos/1/ask", json={"question": "Private question"}, headers={"X-Signed-In-As": "1"})

    assert response.status_code == 401
    assert response.headers["X-Session-Ended"] == "true"
    assert db_session.query(Conversation).count() == 0


def test_a_page_for_another_account_is_refused(alice: TestClient) -> None:
    response = alice.get("/api/finder/searches", headers={"X-Signed-In-As": "99"})

    assert response.status_code == 401


def test_a_page_for_this_account_is_allowed(alice: TestClient) -> None:
    assert alice.get("/api/finder/searches", headers={"X-Signed-In-As": "1"}).status_code == 200


# --- with login not configured, the app is guest-only


@pytest.mark.parametrize("missing", list(LOGIN_SETTINGS))
def test_any_missing_setting_turns_login_off(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.setenv(missing, "")
    browser = visitor()

    assert browser.get("/api/me").json() == {"login_enabled": False, "user": None}
    assert browser.get("/auth/github/login").status_code == 404
    assert browser.get("/auth/github/callback", params={"code": "c", "state": "s"}).status_code == 404


def test_without_login_a_session_cookie_means_nothing(alice: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_CLIENT_ID", "")
    get_settings.cache_clear()

    assert alice.get("/api/me").json() == {"login_enabled": False, "user": None}


def test_login_settings_are_off_by_default() -> None:
    assert visitor().get("/api/me").json() == {"login_enabled": False, "user": None}


# --- data isolation: searches and finder guides


def test_a_new_search_belongs_to_the_user_who_ran_it(
    alice: TestClient, guest: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = RepoCandidate(
        full_name="acme/sample", html_url=REPO_URL, description="About", stars=1, language="Python",
        topics=[], pushed_at=None, license=None,
    )
    assessed = RepoResult(full_name="acme/sample", score=90, why_it_fits="Fits", caveats="None", how_to_start=[])

    async def fake_find(problem: str) -> tuple[SearchPlan, list[Candidate]]:
        return SearchPlan(subject="pdf", queries=["pdf table"]), [Candidate(found, "README")]

    async def fake_rank(problem: str, plan: SearchPlan, candidates: list[Candidate]) -> list[RepoResult]:
        return [assessed]

    monkeypatch.setattr("app.routers.finder.find_repos", fake_find)
    monkeypatch.setattr("app.routers.finder.rank_candidates", fake_rank)

    assert alice.post("/api/finder/search", json={"problem": "Private problem"}).status_code == 201
    assert guest.post("/api/finder/search", json={"problem": "Public problem"}).status_code == 201

    owners = {search.problem: search.owner_user_id for search in db_session.scalars(select(Search))}
    assert owners == {"Private problem": 1, "Public problem": None}
    results = db_session.scalars(select(SearchResult).join(Search).order_by(Search.id)).all()
    assert [result.owner_user_id for result in results] == [1, None]
    assert [item["problem"] for item in alice.get("/api/finder/searches").json()] == ["Private problem"]
    assert [item["problem"] for item in guest.get("/api/finder/searches").json()] == ["Public problem"]


def test_each_space_lists_only_its_own_searches(alice: TestClient, bob: TestClient, guest: TestClient, db_session: Session) -> None:
    save_search(db_session, "Alice's", user_id(db_session, "alice"))
    save_search(db_session, "Bob's", user_id(db_session, "bob"))
    save_search(db_session, "Public", None)

    problems = lambda client: [item["problem"] for item in client.get("/api/finder/searches").json()]  # noqa: E731

    assert problems(alice) == ["Alice's"]
    assert problems(bob) == ["Bob's"]
    assert problems(guest) == ["Public"]


def test_someone_elses_search_answers_404(alice: TestClient, bob: TestClient, guest: TestClient, db_session: Session) -> None:
    search_id, _ = save_search(db_session, "Alice's", user_id(db_session, "alice"))

    assert alice.get(f"/api/finder/searches/{search_id}").json()["results"][0]["problem_guide"] == "A private guide."
    assert bob.get(f"/api/finder/searches/{search_id}").status_code == 404
    assert guest.get(f"/api/finder/searches/{search_id}").status_code == 404


def test_private_and_missing_items_answer_identically(alice: TestClient, guest: TestClient, db_session: Session) -> None:
    search_id, _ = save_search(db_session, "Alice's", user_id(db_session, "alice"))

    private, missing = guest.get(f"/api/finder/searches/{search_id}"), guest.get("/api/finder/searches/999")

    assert (private.status_code, private.json()) == (missing.status_code, missing.json())


def test_a_public_search_is_not_in_a_users_space(alice: TestClient, db_session: Session) -> None:
    search_id, result_id = save_search(db_session, "Public", None)

    assert alice.get(f"/api/finder/searches/{search_id}").status_code == 404
    assert alice.post(f"/api/finder/results/{result_id}/guide").status_code == 404


@pytest.mark.parametrize("suffix", ["index", "guide"])
def test_someone_elses_result_answers_404(
    alice: TestClient, bob: TestClient, guest: TestClient, db_session: Session, suffix: str
) -> None:
    _, result_id = save_search(db_session, "Alice's", user_id(db_session, "alice"))

    assert bob.post(f"/api/finder/results/{result_id}/{suffix}").status_code == 404
    assert guest.post(f"/api/finder/results/{result_id}/{suffix}").status_code == 404
    assert db_session.query(Repo).count() == 0


def test_indexing_a_result_puts_the_repo_in_the_workspace(alice: TestClient, db_session: Session) -> None:
    owner = user_id(db_session, "alice")
    _, result_id = save_search(db_session, "Alice's", owner)

    repo = alice.post(f"/api/finder/results/{result_id}/index").json()

    assert repo["is_public"] is False
    assert db_session.get(UserRepo, (owner, repo["id"])) is not None
    guide = alice.post(f"/api/finder/results/{result_id}/guide")
    assert guide.status_code == 200
    assert guide.json()["repo_id"] == repo["id"]


# --- data isolation: repos


def test_a_users_repo_is_private_to_them(alice: TestClient, bob: TestClient, guest: TestClient, cloned_urls: list[str]) -> None:
    repo = add_repo(alice)

    assert repo["is_public"] is False
    assert [item["id"] for item in alice.get("/api/repos").json()] == [repo["id"]]
    assert bob.get("/api/repos").json() == []
    assert guest.get("/api/repos").json() == []
    assert cloned_urls == [REPO_URL]


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/api/repos/1", None),
        ("get", "/api/repos/1/guide", None),
        ("post", "/api/repos/1/guide", None),
        ("get", "/api/repos/1/conversations", None),
        ("get", "/api/repos/1/file?path=src/app.py", None),
        ("post", "/api/repos/1/ask", {"question": "Hi?"}),
        ("post", "/api/repos/1/reindex", None),
        ("delete", "/api/repos/1", None),
    ],
)
def test_someone_elses_repo_answers_404_everywhere(
    alice: TestClient, bob: TestClient, guest: TestClient, method: str, path: str, body: dict[str, str] | None
) -> None:
    add_repo(alice)

    for client in (bob, guest):
        assert client.request(method, path, json=body).status_code == 404
    assert alice.get("/api/repos/1").json()["status"] == "ready"


def test_a_user_can_read_public_repos(alice: TestClient, guest: TestClient) -> None:
    public = add_repo(guest)  # demo mode is off here, so no password is needed

    assert public["is_public"] is True
    assert [item["id"] for item in alice.get("/api/repos").json()] == [public["id"]]
    assert alice.get(f"/api/repos/{public['id']}/file", params={"path": "src/app.py"}).status_code == 200
    assert ask(alice, public["id"]) == 1


def test_the_list_mixes_public_repos_and_the_users_own(alice: TestClient, bob: TestClient, guest: TestClient) -> None:
    public, own = add_repo(guest), add_repo(alice, OTHER_URL)

    assert {item["id"] for item in alice.get("/api/repos").json()} == {public["id"], own["id"]}
    assert {item["id"] for item in bob.get("/api/repos").json()} == {public["id"]}


def test_a_repo_is_indexed_once_and_shared(
    alice: TestClient, bob: TestClient, cloned_urls: list[str], db_session: Session
) -> None:
    first, second = add_repo(alice), add_repo(bob)

    assert first["id"] == second["id"]
    assert cloned_urls == [REPO_URL]
    assert db_session.query(UserRepo).count() == 2


def test_adding_a_public_repo_links_it_and_keeps_it_public(alice: TestClient, guest: TestClient, cloned_urls: list[str]) -> None:
    add_repo(guest)

    assert add_repo(alice)["is_public"] is True
    assert cloned_urls == [REPO_URL]


def test_adding_twice_is_harmless(alice: TestClient, db_session: Session) -> None:
    add_repo(alice)
    add_repo(alice)

    assert db_session.query(UserRepo).count() == 1


def test_a_shared_repos_chats_stay_separate(alice: TestClient, bob: TestClient) -> None:
    repo = add_repo(alice)["id"]
    add_repo(bob)

    alices = ask(alice, repo, "Alice's question")
    bobs = ask(bob, repo, "Bob's question")

    assert [item["id"] for item in alice.get(f"/api/repos/{repo}/conversations").json()] == [alices]
    assert [item["id"] for item in bob.get(f"/api/repos/{repo}/conversations").json()] == [bobs]


# --- data isolation: conversations


def test_public_chats_and_a_users_chats_are_separate(alice: TestClient, guest: TestClient, db_session: Session) -> None:
    repo = add_repo(guest)["id"]
    public = ask(guest, repo, "Guest question")
    private = ask(alice, repo, "Private question")

    assert [item["id"] for item in guest.get(f"/api/repos/{repo}/conversations").json()] == [public]
    assert [item["id"] for item in alice.get(f"/api/repos/{repo}/conversations").json()] == [private]
    assert db_session.get(Conversation, public).owner_user_id is None
    assert db_session.get(Conversation, private).owner_user_id == user_id(db_session, "alice")


def test_someone_elses_messages_answer_404(alice: TestClient, bob: TestClient, guest: TestClient) -> None:
    repo = add_repo(guest)["id"]
    private = ask(alice, repo, "Private question")
    public = ask(guest, repo, "Public question")

    assert "Private question" in alice.get(f"/api/conversations/{private}/messages").text
    assert bob.get(f"/api/conversations/{private}/messages").status_code == 404
    assert guest.get(f"/api/conversations/{private}/messages").status_code == 404
    assert alice.get(f"/api/conversations/{public}/messages").status_code == 404


def test_nobody_can_continue_someone_elses_conversation(alice: TestClient, bob: TestClient, guest: TestClient) -> None:
    repo = add_repo(guest)["id"]
    private = ask(alice, repo)
    public = ask(guest, repo)
    body = lambda conversation: {"question": "Injected?", "conversation_id": conversation}  # noqa: E731

    assert bob.post(f"/api/repos/{repo}/ask", json=body(private)).status_code == 404
    assert guest.post(f"/api/repos/{repo}/ask", json=body(private)).status_code == 404
    assert alice.post(f"/api/repos/{repo}/ask", json=body(public)).status_code == 404
    assert "Injected" not in alice.get(f"/api/conversations/{private}/messages").text


# --- changing repos


def test_a_user_can_reindex_their_own_repo(alice: TestClient, cloned_urls: list[str]) -> None:
    add_repo(alice)

    assert alice.post("/api/repos/1/reindex").status_code == 202
    assert cloned_urls == [REPO_URL, REPO_URL]


def test_a_user_cannot_change_a_public_repo_without_the_password(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    add_repo(visitor(), headers=PASSWORD)
    alice = sign_in(github)

    assert alice.post("/api/repos/1/reindex").status_code == 401
    assert alice.delete("/api/repos/1").status_code == 401
    assert alice.post("/api/repos/1/reindex", headers=PASSWORD).status_code == 202


def test_linking_a_public_repo_gives_no_right_to_reindex_it(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    add_repo(visitor(), headers=PASSWORD)
    alice = sign_in(github)
    add_repo(alice)

    assert alice.post("/api/repos/1/reindex").status_code == 401


def test_removing_a_public_repo_only_removes_the_link(alice: TestClient, guest: TestClient, db_session: Session) -> None:
    add_repo(guest)
    add_repo(alice)
    ask(alice, 1, "Private question")
    public_chat = ask(guest, 1, "Public question")

    assert alice.delete("/api/repos/1").status_code == 204

    assert guest.get("/api/repos/1").status_code == 200
    assert db_session.query(UserRepo).count() == 0
    assert [item.id for item in db_session.scalars(select(Conversation))] == [public_chat]


def test_removing_a_shared_private_repo_leaves_it_for_the_other_user(alice: TestClient, bob: TestClient) -> None:
    add_repo(alice)
    add_repo(bob)
    bobs_chat = ask(bob, 1)
    ask(alice, 1)

    assert alice.delete("/api/repos/1").status_code == 204

    assert alice.get("/api/repos/1").status_code == 404
    assert bob.get("/api/repos/1").json()["status"] == "ready"
    assert [item["id"] for item in bob.get("/api/repos/1/conversations").json()] == [bobs_chat]


def test_the_last_user_removing_a_private_repo_deletes_it(alice: TestClient, db_session: Session) -> None:
    add_repo(alice)
    ask(alice, 1)

    assert alice.delete("/api/repos/1").status_code == 204

    assert db_session.query(Repo).count() == 0
    assert db_session.query(Conversation).count() == 0


def test_removing_a_repo_unlinks_the_users_results_but_keeps_their_guides(alice: TestClient, bob: TestClient, db_session: Session) -> None:
    _, result_id = save_search(db_session, "Alice's", user_id(db_session, "alice"))
    alice.post(f"/api/finder/results/{result_id}/index")
    add_repo(bob)  # keeps the repo alive after Alice removes it

    alice.delete("/api/repos/1")

    db_session.expire_all()
    result = db_session.get(SearchResult, result_id)
    assert result.repo_id is None
    assert result.problem_guide == "A private guide."
    assert alice.post(f"/api/finder/results/{result_id}/guide").status_code == 409


def test_a_guest_who_adds_a_private_repo_needs_the_password_and_learns_nothing(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    alice = sign_in(github)
    add_repo(alice)
    guest = visitor()

    response = guest.post("/api/repos", json={"url": REPO_URL})
    new = guest.post("/api/repos", json={"url": OTHER_URL})

    assert response.status_code == new.status_code == 401
    assert response.json() == new.json()
    assert guest.get("/api/repos").json() == []


def test_the_password_holder_can_make_a_users_repo_public_without_reindexing(
    demo: pytest.MonkeyPatch, github: FakeGitHub, cloned_urls: list[str]
) -> None:
    alice = sign_in(github)
    add_repo(alice)
    ask(alice, 1, "Private question")
    guest = visitor()

    assert guest.post("/api/repos", json={"url": REPO_URL}, headers=PASSWORD).status_code == 200

    assert cloned_urls == [REPO_URL]
    assert [repo["id"] for repo in guest.get("/api/repos").json()] == [1]
    assert guest.get("/api/repos/1/conversations").json() == []


# --- what happens to data


def test_deleting_a_user_never_publishes_their_private_rows(alice: TestClient, guest: TestClient, db_session: Session) -> None:
    repo = add_repo(guest)["id"]
    owner = user_id(db_session, "alice")
    ask(alice, repo, "Private question")
    save_search(db_session, "Alice's", owner)

    db_session.execute(text("DELETE FROM users"))
    db_session.commit()

    assert db_session.query(Search).count() == 0
    assert db_session.query(SearchResult).count() == 0
    assert db_session.query(Conversation).count() == 0
    assert guest.get("/api/finder/searches").json() == []


def test_existing_rows_stay_public(guest: TestClient, db_session: Session) -> None:
    # What the migration leaves behind: rows with no owner, and repos that default to public.
    db_session.execute(text("INSERT INTO repos (url, name) VALUES ('https://github.com/old/repo', 'old/repo')"))
    db_session.execute(text("""INSERT INTO searches (problem, plan) VALUES ('Old search', '{}'::jsonb)"""))
    db_session.commit()

    assert [repo["name"] for repo in guest.get("/api/repos").json()] == ["old/repo"]
    assert [search["problem"] for search in guest.get("/api/finder/searches").json()] == ["Old search"]


# --- limits


def test_signed_in_users_index_without_the_password(demo: pytest.MonkeyPatch, github: FakeGitHub, cloned_urls: list[str]) -> None:
    add_repo(sign_in(github))

    assert cloned_urls == [REPO_URL]


def test_the_size_limit_applies_to_signed_in_users(demo: pytest.MonkeyPatch, github: FakeGitHub, db_session: Session) -> None:
    demo.setattr("app.demo._repo_size_mb", lambda name: 500.0)

    response = sign_in(github).post("/api/repos", json={"url": REPO_URL})

    assert response.status_code == 413
    assert db_session.query(Repo).count() == 0
    assert db_session.query(UserDailyUsage).count() == 0  # a refused repo doesn't use up a slot


def test_repos_per_day_is_per_user(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    demo.setenv("USER_REPOS_PER_DAY", "1")
    demo.setenv("SIGNED_IN_CHUNKS_PER_DAY", "1000")
    alice, bob = sign_in(github), sign_in(github, BOB)
    add_repo(alice)

    response = alice.post("/api/repos", json={"url": OTHER_URL})

    assert response.status_code == 429
    assert "indexed repos a day" in response.json()["detail"]
    add_repo(bob, OTHER_URL)


def test_adding_a_repo_already_here_costs_no_slot(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    demo.setenv("USER_REPOS_PER_DAY", "1")
    demo.setenv("SIGNED_IN_CHUNKS_PER_DAY", "1000")
    alice, bob = sign_in(github), sign_in(github, BOB)
    add_repo(bob)
    add_repo(alice)

    add_repo(alice, OTHER_URL)


def test_reindexing_uses_up_a_slot(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    demo.setenv("USER_REPOS_PER_DAY", "1")
    demo.setenv("SIGNED_IN_CHUNKS_PER_DAY", "1000")
    alice = sign_in(github)
    add_repo(alice)

    assert alice.post("/api/repos/1/reindex").status_code == 429


def test_signed_in_indexing_stops_at_the_shared_chunk_budget(demo: pytest.MonkeyPatch, github: FakeGitHub, db_session: Session) -> None:
    demo.setenv("SIGNED_IN_CHUNKS_PER_DAY", "1")

    add_repo(sign_in(github))

    repo = db_session.scalars(select(Repo)).one()
    assert repo.status == "failed"
    assert "indexing budget" in repo.error
    assert repo.chunks_embedded == 0


def test_the_chunk_budget_is_shared_between_users(demo: pytest.MonkeyPatch, github: FakeGitHub, db_session: Session) -> None:
    add_repo(sign_in(github))
    used = db_session.scalars(select(DailyUsage)).one().user_chunks
    demo.setenv("SIGNED_IN_CHUNKS_PER_DAY", str(used + 1))
    get_settings.cache_clear()

    add_repo(sign_in(github, BOB), OTHER_URL)

    assert used > 0
    assert db_session.get(Repo, 1).status == "ready"
    assert db_session.get(Repo, 2).status == "failed"


def test_a_guests_repo_is_not_charged_to_the_chunk_budget(demo: pytest.MonkeyPatch, db_session: Session) -> None:
    demo.setenv("SIGNED_IN_CHUNKS_PER_DAY", "1")

    add_repo(visitor(), headers=PASSWORD)

    assert db_session.scalars(select(Repo)).one().status == "ready"


def test_rate_limits_are_per_user_even_from_one_ip(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    demo.setenv("RATE_LIMIT_ASK_PER_HOUR", "1")
    add_repo(visitor(), headers=PASSWORD)
    alice, bob = sign_in(github), sign_in(github, BOB)
    question: Callable[[TestClient], int] = lambda client: client.post("/api/repos/1/ask", json={"question": "Hi?"}).status_code

    assert [question(alice), question(alice), question(bob)] == [200, 429, 200]


def test_a_user_and_a_guest_on_one_ip_have_separate_limits(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    demo.setenv("RATE_LIMIT_ASK_PER_HOUR", "1")
    guest = visitor()
    add_repo(guest, headers=PASSWORD)
    alice = sign_in(github)
    question: Callable[[TestClient], int] = lambda client: client.post("/api/repos/1/ask", json={"question": "Hi?"}).status_code

    assert [question(guest), question(alice), question(guest)] == [200, 200, 429]


def test_the_daily_cap_is_per_user(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    demo.setenv("USER_DAILY_REQUEST_CAP", "1")
    add_repo(visitor(), headers=PASSWORD)
    alice, bob = sign_in(github), sign_in(github, BOB)
    question: Callable[[TestClient], httpx.Response] = lambda client: client.post("/api/repos/1/ask", json={"question": "Hi?"})

    assert question(alice).status_code == 200
    refused = question(alice)
    assert refused.status_code == 429
    assert "your 1 requests for today" in refused.json()["detail"]
    assert question(bob).status_code == 200


def test_signed_in_users_share_a_total_cap(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    demo.setenv("SIGNED_IN_DAILY_REQUEST_CAP", "1")
    add_repo(visitor(), headers=PASSWORD)
    alice, bob = sign_in(github), sign_in(github, BOB)
    question: Callable[[TestClient], int] = lambda client: client.post("/api/repos/1/ask", json={"question": "Hi?"}).status_code

    assert [question(alice), question(bob)] == [200, 429]


def test_signed_in_requests_do_not_use_up_the_guests_cap(demo: pytest.MonkeyPatch, github: FakeGitHub) -> None:
    demo.setenv("DAILY_REQUEST_CAP", "1")
    guest = visitor()
    add_repo(guest, headers=PASSWORD)
    alice = sign_in(github)

    for _ in range(3):
        assert alice.post("/api/repos/1/ask", json={"question": "Hi?"}).status_code == 200

    assert guest.post("/api/repos/1/ask", json={"question": "Hi?"}).status_code == 200
    refused = guest.post("/api/repos/1/ask", json={"question": "Hi?"})
    assert refused.status_code == 429
    assert refused.json()["detail"].startswith("Demo limit reached: this public demo handles 1 requests")


# --- guests are unchanged when login is on


def test_guests_still_need_the_password_to_add_a_repo(demo: pytest.MonkeyPatch) -> None:
    response = visitor().post("/api/repos", json={"url": REPO_URL})

    assert response.status_code == 401
    assert "index password" in response.json()["detail"]


def test_guests_keep_the_shared_public_space(demo: pytest.MonkeyPatch) -> None:
    first, second = visitor(), visitor()
    add_repo(first, headers=PASSWORD)

    conversation = ask(first, 1, "A public question")

    assert add_repo(second)["id"] == 1
    assert [item["id"] for item in second.get("/api/repos/1/conversations").json()] == [conversation]
    assert "A public question" in second.get(f"/api/conversations/{conversation}/messages").text


def test_guests_keep_their_per_ip_limit(demo: pytest.MonkeyPatch) -> None:
    demo.setenv("RATE_LIMIT_ASK_PER_HOUR", "1")
    demo.setenv("CLIENT_IP_HEADER", "true-client-ip")
    guest = visitor()
    add_repo(guest, headers=PASSWORD)
    question = lambda ip: guest.post("/api/repos/1/ask", json={"question": "Hi?"}, headers={"True-Client-IP": ip}).status_code  # noqa: E731

    assert [question("1.1.1.1"), question("1.1.1.1"), question("2.2.2.2")] == [200, 429, 200]


def test_a_guests_request_looks_nobody_up_in_the_database(guest: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise AssertionError("A guest's request shouldn't look anyone up.")

    monkeypatch.setattr("app.auth.SessionLocal", fail)

    assert guest.get("/api/me").json()["user"] is None
