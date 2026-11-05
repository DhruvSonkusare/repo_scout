"""Tests for the Repo Finder: fake LLM replies, a mocked GitHub and the real Postgres database."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai.errors import ClientError, ServerError
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from sqlalchemy.orm import Session

from app.db import Search, SearchResult
from app.finder.chains import (
    MAX_CANDIDATES,
    MAX_README_CHARS_FOR_RERANK,
    PLANNER_PROMPT,
    Candidate,
    Ranking,
    RepoAssessment,
    SearchPlan,
    _clean_ranking,
    _format_candidates,
    _merge_candidates,
    find_repos,
    rank_candidates,
)
from app.finder.github import GitHubClient, RepoCandidate
from app.main import app

pytestmark = pytest.mark.usefixtures("fresh_settings")

client = TestClient(app)
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)
RECENT = datetime.now(timezone.utc) - timedelta(days=30)


def repo(name: str = "acme/tables", stars: int = 100, pushed_at: datetime | None = RECENT) -> RepoCandidate:
    return RepoCandidate(
        full_name=name,
        html_url=f"https://github.com/{name}",
        description=f"About {name}",
        stars=stars,
        language="Python",
        topics=["pdf"],
        pushed_at=pushed_at,
        license="MIT",
    )


def github_item(name: str, stars: int = 100, pushed_at: datetime | None = RECENT) -> dict[str, Any]:
    return {
        "full_name": name,
        "html_url": f"https://github.com/{name}",
        "description": f"About {name}",
        "stargazers_count": stars,
        "language": "Python",
        "topics": ["pdf"],
        "pushed_at": pushed_at.isoformat() if pushed_at else None,
        "license": {"spdx_id": "MIT"},
    }


class FakeGitHub:
    """A fake GitHub that returns scripted repos per search query."""

    def __init__(self) -> None:
        self.items_by_query: dict[str, list[dict[str, Any]]] = {}
        self.status_by_query: dict[str, int] = {}
        self.readme_status: dict[str, int] = {}
        self.queries: list[str] = []
        self.readmes_read: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search/repositories":
            full_query = request.url.params["q"]
            self.queries.append(full_query)
            query = full_query.split(" language:")[0].removesuffix(" archived:false")
            if query in self.status_by_query:
                return httpx.Response(self.status_by_query[query], json={"message": "test"})
            return httpx.Response(200, json={"items": self.items_by_query.get(query, [])})
        name = request.url.path.removeprefix("/repos/").removesuffix("/readme")
        self.readmes_read.append(name)
        status = self.readme_status.get(name, 200)
        return httpx.Response(status, text=f"README of {name}" if status == 200 else "error")


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> FakeGitHub:
    """Point the finder at a fake GitHub."""
    fake = FakeGitHub()
    transport = httpx.MockTransport(fake.handler)
    monkeypatch.setattr("app.finder.chains.GitHubClient", partial(GitHubClient, transport=transport))
    return fake


class FakeLLM:
    """Replaces get_structured_model: replies by schema, or raises, and records the prompts."""

    def __init__(self) -> None:
        self.plan = SearchPlan(queries=["pdf table", "tabular pdf"], language=None, must_have=["reads tables"])
        self.ranking: Ranking | None = None
        self.errors: dict[type, Exception] = {}
        self.prompts: dict[type, list[BaseMessage]] = {}

    def get_model(self, schema: type, fake_reply: Callable[[Any], Any] | None = None) -> Runnable:
        def reply(prompt: PromptValue) -> Any:
            self.prompts[schema] = prompt.to_messages()
            if schema in self.errors:
                raise self.errors[schema]
            if schema is SearchPlan:
                return self.plan
            return self.ranking if self.ranking is not None else fake_reply(prompt)

        return RunnableLambda(reply)


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Script the planner and reranker replies."""
    fake = FakeLLM()
    monkeypatch.setattr("app.finder.chains.get_structured_model", fake.get_model)
    return fake


def assessment(name: str, score: int = 80, steps: int = 2) -> RepoAssessment:
    return RepoAssessment(
        full_name=name,
        score=score,
        why_it_fits=f"{name} fits",
        caveats="None noted",
        how_to_start=[f"step {number}" for number in range(steps)],
    )


def candidate(name: str = "acme/tables", readme: str = "A README") -> Candidate:
    return Candidate(repo(name), readme)


def search(problem: str = "Extract tables from PDFs") -> httpx.Response:
    return client.post("/api/finder/search", json={"problem": problem})


# --- the planner prompt


def test_planner_prompt_asks_for_broad_queries_with_synonyms() -> None:
    system = PLANNER_PROMPT.messages[0].prompt.template

    assert "EVERY word" in system
    assert "2 words each" in system
    assert "synonym" in system
    assert "Every query keeps the problem's main subject" in system
    assert "Never put a programming language in a query" in system


# --- finding candidates


def test_merge_removes_duplicates_across_queries_and_ignores_case() -> None:
    merged = _merge_candidates([[repo("acme/a"), repo("acme/b")], [repo("ACME/A"), repo("acme/c")]], NOW)

    assert sorted(item.full_name.lower() for item in merged) == ["acme/a", "acme/b", "acme/c"]


def test_merge_drops_repos_with_no_commits_in_three_years() -> None:
    old = NOW - timedelta(days=3 * 365 + 1)
    recent_enough = NOW - timedelta(days=3 * 365 - 1)
    results = [[repo("acme/old", pushed_at=old), repo("acme/kept", pushed_at=recent_enough), repo("acme/never", pushed_at=None)]]

    assert [item.full_name for item in _merge_candidates(results, NOW)] == ["acme/kept"]


def test_merge_keeps_the_most_starred_up_to_the_limit() -> None:
    repos = [repo(f"acme/{number}", stars=number) for number in range(MAX_CANDIDATES + 10)]

    merged = _merge_candidates([repos], NOW)

    assert len(merged) == MAX_CANDIDATES
    assert merged[0].stars == MAX_CANDIDATES + 9
    assert merged[-1].stars == 10


def test_find_repos_searches_every_query_and_reads_each_readme_once(github: FakeGitHub, llm: FakeLLM) -> None:
    github.items_by_query = {
        "pdf table": [github_item("acme/a"), github_item("acme/b")],
        "tabular pdf": [github_item("acme/b"), github_item("acme/c")],
    }

    plan, candidates = asyncio.run(find_repos("Extract tables"))

    assert plan == llm.plan
    assert sorted(github.queries) == ["pdf table archived:false", "tabular pdf archived:false"]
    assert sorted(item.repo.full_name for item in candidates) == ["acme/a", "acme/b", "acme/c"]
    assert sorted(github.readmes_read) == ["acme/a", "acme/b", "acme/c"]
    assert candidates[0].readme.startswith("README of")


def test_the_planned_language_filters_every_query(github: FakeGitHub, llm: FakeLLM) -> None:
    llm.plan = SearchPlan(queries=["whatsapp bot", "whatsapp api"], language="Python", must_have=[])

    asyncio.run(find_repos("WhatsApp bot in Python"))

    assert all("language:Python" in query for query in github.queries)


def test_only_three_queries_are_run_and_blank_ones_are_skipped(github: FakeGitHub, llm: FakeLLM) -> None:
    llm.plan = SearchPlan(queries=["one", "  ", "two", "three", "four"], language=None, must_have=[])

    asyncio.run(find_repos("Anything"))

    assert sorted(github.queries) == ["one archived:false", "three archived:false", "two archived:false"]


def test_readmes_are_trimmed_for_the_reranker(
    github: FakeGitHub, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/a")]
    long_readme = httpx.Response(200, text="x" * (MAX_README_CHARS_FOR_RERANK + 500))
    original = github.handler
    transport = httpx.MockTransport(
        lambda request: long_readme if request.url.path.endswith("/readme") else original(request)
    )
    monkeypatch.setattr("app.finder.chains.GitHubClient", partial(GitHubClient, transport=transport))

    _, candidates = asyncio.run(find_repos("Anything"))

    assert len(candidates[0].readme) == MAX_README_CHARS_FOR_RERANK


def test_one_failed_query_is_fine_if_another_works(
    github: FakeGitHub, llm: FakeLLM, caplog: pytest.LogCaptureFixture
) -> None:
    github.items_by_query["tabular pdf"] = [github_item("acme/a")]
    github.status_by_query["pdf table"] = 500

    _, candidates = asyncio.run(find_repos("Anything"))

    assert [item.repo.full_name for item in candidates] == ["acme/a"]
    assert "A GitHub search failed" in caplog.text


def test_a_failed_readme_does_not_lose_the_candidate(
    github: FakeGitHub, llm: FakeLLM, caplog: pytest.LogCaptureFixture
) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/a")]
    github.readme_status["acme/a"] = 500

    _, candidates = asyncio.run(find_repos("Anything"))

    assert candidates[0].repo.full_name == "acme/a"
    assert candidates[0].readme == ""
    assert "Couldn't read the README of acme/a" in caplog.text


# --- ranking


def test_format_describes_each_candidate() -> None:
    text = _format_candidates([candidate("acme/a", "Reads tables")])

    assert "### acme/a" in text
    assert "Stars: 100. Language: Python. License: MIT." in text
    assert "Description: About acme/a" in text
    assert "Topics: pdf" in text
    assert "Reads tables" in text


def test_format_says_when_there_is_no_readme() -> None:
    assert "(no README)" in _format_candidates([candidate("acme/a", "")])


def test_clean_drops_names_that_are_not_candidates(caplog: pytest.LogCaptureFixture) -> None:
    ranking = Ranking(results=[assessment("acme/a"), assessment("invented/repo")])

    cleaned = _clean_ranking(ranking, [candidate("acme/a")])

    assert [item.full_name for item in cleaned] == ["acme/a"]
    assert "invented/repo" in caplog.text


def test_clean_uses_githubs_spelling_of_the_name() -> None:
    cleaned = _clean_ranking(Ranking(results=[assessment(" ACME/a ")]), [candidate("acme/A")])

    assert cleaned[0].full_name == "acme/A"


def test_clean_keeps_the_first_of_a_repeated_repo() -> None:
    ranking = Ranking(results=[assessment("acme/a", 70), assessment("acme/a", 95)])

    assert [item.score for item in _clean_ranking(ranking, [candidate("acme/a")])] == [70]


def test_clean_sorts_best_first() -> None:
    ranking = Ranking(results=[assessment("acme/a", 30), assessment("acme/b", 90), assessment("acme/c", 60)])

    cleaned = _clean_ranking(ranking, [candidate("acme/a"), candidate("acme/b"), candidate("acme/c")])

    assert [item.full_name for item in cleaned] == ["acme/b", "acme/c", "acme/a"]


@pytest.mark.parametrize(("score", "expected"), [(-5, 0), (0, 0), (100, 100), (250, 100)])
def test_clean_keeps_scores_in_range(score: int, expected: int) -> None:
    cleaned = _clean_ranking(Ranking(results=[assessment("acme/a", score)]), [candidate("acme/a")])

    assert cleaned[0].score == expected


def test_clean_limits_the_start_steps() -> None:
    cleaned = _clean_ranking(Ranking(results=[assessment("acme/a", steps=7)]), [candidate("acme/a")])

    assert len(cleaned[0].how_to_start) == 4


def test_no_candidates_skips_the_reranker_call(llm: FakeLLM) -> None:
    assert asyncio.run(rank_candidates("Anything", llm.plan, [])) == []
    assert Ranking not in llm.prompts


def test_reranker_prompt_has_the_problem_requirements_and_rules(llm: FakeLLM) -> None:
    asyncio.run(rank_candidates("Extract tables", llm.plan, [candidate("acme/a", "Reads tables")]))

    system, human = llm.prompts[Ranking]
    assert human.content == "My problem: Extract tables"
    assert "reads tables" in system.content
    assert "Reads tables" in system.content
    assert "never invent features" in system.content
    assert "untrusted" in system.content
    assert "low scores for everything is a correct answer" in system.content


def test_braces_in_a_readme_do_not_break_the_prompt(llm: FakeLLM) -> None:
    results = asyncio.run(rank_candidates("Anything", llm.plan, [candidate("acme/a", "Use {problem} and {oops}")]))

    assert "Use {problem} and {oops}" in llm.prompts[Ranking][0].content
    assert len(results) == 1


# --- the fake provider, with nothing replaced except GitHub


def test_the_fake_provider_returns_a_fixed_plan_and_ranking(github: FakeGitHub) -> None:
    github.items_by_query["fake query"] = [github_item("acme/a", stars=5), github_item("acme/b", stars=9)]

    response = search()

    assert response.status_code == 201
    body = response.json()
    assert body["plan"]["queries"] == ["fake query"]
    assert sorted(result["full_name"] for result in body["results"]) == ["acme/a", "acme/b"]
    assert all(result["why_it_fits"].startswith("A fake assessment") for result in body["results"])


# --- POST /api/finder/search


def test_search_saves_and_returns_results_sorted_by_score(github: FakeGitHub, llm: FakeLLM, db_session: Session) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/low", stars=9000), github_item("acme/high", stars=5)]
    llm.ranking = Ranking(results=[assessment("acme/low", 20), assessment("acme/high", 95)])

    response = search("Extract tables from PDFs")

    assert response.status_code == 201
    body = response.json()
    assert body["problem"] == "Extract tables from PDFs"
    assert body["plan"] == {"queries": ["pdf table", "tabular pdf"], "language": None, "must_have": ["reads tables"]}
    assert [result["full_name"] for result in body["results"]] == ["acme/high", "acme/low"]
    first = body["results"][0]
    assert first["url"] == "https://github.com/acme/high"
    assert first["description"] == "About acme/high"
    assert first["stars"] == 5
    assert first["language"] == "Python"
    assert first["last_pushed"] is not None
    assert first["score"] == 95
    assert first["why_it_fits"] == "acme/high fits"
    assert first["caveats"] == "None noted"
    assert first["how_to_start"] == ["step 0", "step 1"]
    assert first["repo_id"] is None
    assert first["problem_guide"] is None
    assert db_session.query(Search).count() == 1
    assert db_session.query(SearchResult).count() == 2


def test_an_invented_repo_is_not_saved(github: FakeGitHub, llm: FakeLLM, db_session: Session) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/real")]
    llm.ranking = Ranking(results=[assessment("acme/real"), assessment("invented/repo", 100)])

    body = search().json()

    assert [result["full_name"] for result in body["results"]] == ["acme/real"]
    assert db_session.query(SearchResult).count() == 1


def test_a_search_with_no_candidates_is_saved_with_no_results(github: FakeGitHub, llm: FakeLLM) -> None:
    response = search()

    assert response.status_code == 201
    assert response.json()["results"] == []
    assert Ranking not in llm.prompts


@pytest.mark.parametrize("problem", ["", "  ", "ab", "x" * 2001])
def test_invalid_problems_return_422(problem: str) -> None:
    assert search(problem).status_code == 422


# --- errors, none of which save anything


def assert_nothing_saved(db_session: Session) -> None:
    assert db_session.query(Search).count() == 0


def test_github_rate_limit_returns_429(github: FakeGitHub, llm: FakeLLM, db_session: Session) -> None:
    github.status_by_query = {"pdf table": 429, "tabular pdf": 429}

    response = search()

    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"]
    assert_nothing_saved(db_session)


def test_every_query_failing_returns_502(github: FakeGitHub, llm: FakeLLM, db_session: Session) -> None:
    github.status_by_query = {"pdf table": 500, "tabular pdf": 500}

    response = search()

    assert response.status_code == 502
    assert "GitHub returned an error" in response.json()["detail"]
    assert_nothing_saved(db_session)


@pytest.mark.parametrize("schema", [SearchPlan, Ranking])
def test_an_unavailable_model_returns_503(
    github: FakeGitHub, llm: FakeLLM, db_session: Session, schema: type
) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/a")]
    llm.errors[schema] = ServerError(503, {"error": {"code": 503, "message": "test"}})

    response = search()

    assert response.status_code == 503
    assert "busy" in response.json()["detail"]
    assert_nothing_saved(db_session)


def test_a_rejected_request_returns_503_with_the_reason(github: FakeGitHub, llm: FakeLLM) -> None:
    llm.errors[SearchPlan] = ClientError(400, {"error": {"code": 400, "message": "test"}})

    response = search()

    assert response.status_code == 503
    assert "rejected the request" in response.json()["detail"]


def test_a_reply_that_cannot_be_parsed_returns_502(github: FakeGitHub, llm: FakeLLM, db_session: Session) -> None:
    llm.errors[SearchPlan] = OutputParserException("not JSON")

    response = search()

    assert response.status_code == 502
    assert "couldn't be read" in response.json()["detail"]
    assert_nothing_saved(db_session)


def test_chains_with_no_structured_model_explain_themselves(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder-key")
    monkeypatch.setenv("CHAT_MODELS", "gemma-4-31b-it")

    response = search()

    assert response.status_code == 503
    assert "supports structured output" in response.json()["detail"]


# --- the real structured chain, with a scripted fake model behind it


class StructuredFake(FakeListChatModel):
    """A fake chat model with structured output, which fails or replies as scripted."""

    label: str
    calls: Any  # not list[str], because pydantic would copy the list the test reads
    failing: dict[str, Exception]

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        def reply(_: Any) -> Any:
            self.calls.append(self.label)
            if self.label in self.failing:
                raise self.failing[self.label]
            return SearchPlan(queries=["from fallback"])

        return RunnableLambda(reply)


def test_the_planner_falls_back_through_the_structured_chain(
    github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    failing: dict[str, Exception] = {"gemini-3-a": ServerError(503, {"error": {"code": 503, "message": "test"}})}
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder-key")
    monkeypatch.setenv("CHAT_MODELS", "gemini-3-a,gemma-4-31b-it,gemini-3-b")

    def build(name: str, is_last: bool, thinking_level: str) -> StructuredFake:
        return StructuredFake(responses=["unused"], label=name, calls=calls, failing=failing)

    monkeypatch.setattr("app.llm._build_chat_model", build)

    plan, _ = asyncio.run(find_repos("Anything"))

    assert plan.queries == ["from fallback"]
    assert calls == ["gemini-3-a", "gemini-3-b"]  # Gemma can't do structured output, so it's left out


# --- past searches


def test_searches_are_listed_newest_first_without_results(github: FakeGitHub, llm: FakeLLM) -> None:
    first = search("First problem").json()["id"]
    second = search("Second problem").json()["id"]

    listed = client.get("/api/finder/searches").json()

    assert [item["id"] for item in listed] == [second, first]
    assert listed[0]["problem"] == "Second problem"
    assert "results" not in listed[0]


def test_a_past_search_returns_its_results_sorted_by_score(github: FakeGitHub, llm: FakeLLM) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/a"), github_item("acme/b")]
    llm.ranking = Ranking(results=[assessment("acme/a", 40), assessment("acme/b", 75)])
    created = search().json()

    response = client.get(f"/api/finder/searches/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created
    assert [result["score"] for result in response.json()["results"]] == [75, 40]


def test_an_unknown_search_returns_404() -> None:
    assert client.get("/api/finder/searches/999").status_code == 404
