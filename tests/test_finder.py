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
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Repo, Search, SearchResult
from app.finder.chains import (
    MAX_CANDIDATES,
    MAX_README_CHARS_FOR_RERANK,
    MIN_CANDIDATES,
    NOT_ASSESSED_REASON,
    RERANK_BATCHES,
    PLANNER_PROMPT,
    PROBLEM_EXCERPT_K,
    PROBLEM_GUIDE_CODE_PROBE_K,
    PROBLEM_GUIDE_CODE_PROBES,
    PROBLEM_GUIDE_PROBE_K,
    PROBLEM_GUIDE_PROBES,
    Candidate,
    Ranking,
    RepoAssessment,
    SearchPlan,
    VerifiedGuide,
    _clean_ranking,
    _format_candidates,
    _merge_candidates,
    _with_subject,
    find_repos,
    rank_candidates,
)
from app.finder.github import GitHubClient, RepoCandidate
from app.llm import FAKE_CHAT_RESPONSES, get_structured_guide_model
from app.main import app
from app.rag import FILTERED_FETCH_MULTIPLIER, collect_excerpts
from tests.conftest import RecordingChatModel

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
        self.plan = SearchPlan(
            subject="pdf", queries=["pdf table", "tabular pdf"], language=None, must_have=["reads tables"]
        )
        self.ranking: Ranking | None = None
        self.failing_repos: dict[str, Exception] = {}  # a batch containing this repo raises this
        self.ranking_prompts: list[str] = []  # the system prompt of each reranker call
        self.errors: dict[type, Exception] = {}
        self.prompts: dict[type, list[BaseMessage]] = {}

    def get_model(self, schema: type, fake_reply: Callable[[Any], Any] | None = None) -> Runnable:
        def reply(prompt: PromptValue) -> Any:
            self.prompts[schema] = prompt.to_messages()
            if schema in self.errors:
                raise self.errors[schema]
            if schema is SearchPlan:
                return self.plan
            system = prompt.to_messages()[0].content
            self.ranking_prompts.append(system)
            for name, error in self.failing_repos.items():
                if f"### {name}\n" in system:
                    raise error
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
    assert "Every query contains the subject" in system
    assert "Never put a programming language in a query" in system


# --- finding candidates


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("pdf table", "pdf table"),
        ("PDF tabular", "PDF tabular"),
        ("pdfs table", "pdfs table"),
        ("table parser", "pdf table parser"),
        ("table parser extra words", "pdf table parser"),
    ],
)
def test_every_query_ends_up_with_the_subject(query: str, expected: str) -> None:
    plan = _with_subject(SearchPlan(subject="pdf", queries=[query]))

    assert plan.queries == [expected]


def test_a_plural_or_capitalized_subject_is_still_recognized() -> None:
    plan = _with_subject(SearchPlan(subject=" PDFs ", queries=["pdf table", "table parser"]))

    assert plan.subject == "pdfs"
    assert plan.queries == ["pdf table", "pdfs table parser"]


def test_repeated_queries_are_removed_after_adding_the_subject() -> None:
    plan = _with_subject(SearchPlan(subject="pdf", queries=["table", "pdf table"]))

    assert plan.queries == ["pdf table"]


def test_a_plan_without_a_subject_is_left_alone(caplog: pytest.LogCaptureFixture) -> None:
    plan = SearchPlan(subject="  ", queries=["table parser"])

    assert _with_subject(plan).queries == ["table parser"]
    assert "gave no subject" in caplog.text


def test_the_planned_queries_are_corrected_and_saved_that_way(github: FakeGitHub, llm: FakeLLM) -> None:
    llm.plan = SearchPlan(subject="pdf", queries=["table extraction", "pdf tabular"], must_have=[])

    body = search().json()

    assert body["plan"]["queries"] == ["pdf table extraction", "pdf tabular"]
    assert "pdf table extraction archived:false" in github.queries


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
    assert sorted(github.queries) == ["pdf archived:false", "pdf table archived:false", "tabular pdf archived:false"]
    assert sorted(item.repo.full_name for item in candidates) == ["acme/a", "acme/b", "acme/c"]
    assert sorted(github.readmes_read) == ["acme/a", "acme/b", "acme/c"]
    assert candidates[0].readme.startswith("README of")


def test_the_planned_language_filters_every_query(github: FakeGitHub, llm: FakeLLM) -> None:
    llm.plan = SearchPlan(subject="whatsapp", queries=["whatsapp bot", "whatsapp api"], language="Python", must_have=[])

    asyncio.run(find_repos("WhatsApp bot in Python"))

    assert all("language:Python" in query for query in github.queries)


def test_only_three_queries_are_run_and_blank_ones_are_skipped(github: FakeGitHub, llm: FakeLLM) -> None:
    llm.plan = SearchPlan(subject="x", queries=["x one", "  ", "x two", "x three", "x four"], must_have=[])
    github.items_by_query["x one"] = [github_item(f"acme/{number}") for number in range(MIN_CANDIDATES)]

    asyncio.run(find_repos("Anything"))

    assert sorted(github.queries) == ["x one archived:false", "x three archived:false", "x two archived:false"]


# --- the broader retry


def test_too_few_candidates_runs_one_search_for_the_subject_alone(github: FakeGitHub, llm: FakeLLM) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/a")]
    github.items_by_query["pdf"] = [github_item("acme/a"), github_item("acme/broad")]

    _, candidates = asyncio.run(find_repos("Anything"))

    assert github.queries.count("pdf archived:false") == 1
    assert len(github.queries) == 3
    assert sorted(item.repo.full_name for item in candidates) == ["acme/a", "acme/broad"]


def test_the_broader_search_keeps_the_language(github: FakeGitHub, llm: FakeLLM) -> None:
    llm.plan = SearchPlan(subject="pdf", queries=["pdf table"], language="Python", must_have=[])

    asyncio.run(find_repos("Anything"))

    assert all("language:Python" in query for query in github.queries)
    assert len(github.queries) == 2


def test_enough_candidates_skips_the_broader_search(github: FakeGitHub, llm: FakeLLM) -> None:
    github.items_by_query["pdf table"] = [github_item(f"acme/{number}") for number in range(MIN_CANDIDATES)]

    asyncio.run(find_repos("Anything"))

    assert "pdf archived:false" not in github.queries


def test_stale_repos_do_not_count_towards_enough_candidates(github: FakeGitHub, llm: FakeLLM) -> None:
    stale = datetime.now(timezone.utc) - timedelta(days=4 * 365)
    github.items_by_query["pdf table"] = [github_item(f"acme/{n}", pushed_at=stale) for n in range(MIN_CANDIDATES)]

    asyncio.run(find_repos("Anything"))

    assert "pdf archived:false" in github.queries


def test_a_failed_broader_search_keeps_what_was_found(
    github: FakeGitHub, llm: FakeLLM, caplog: pytest.LogCaptureFixture
) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/a")]
    github.status_by_query["pdf"] = 429

    _, candidates = asyncio.run(find_repos("Anything"))

    assert [item.repo.full_name for item in candidates] == ["acme/a"]
    assert "broader GitHub search failed" in caplog.text


def test_the_broader_search_is_skipped_if_the_plan_already_searched_the_subject(
    github: FakeGitHub, llm: FakeLLM
) -> None:
    llm.plan = SearchPlan(subject="pdf", queries=["pdf"], must_have=[])

    asyncio.run(find_repos("Anything"))

    assert github.queries == ["pdf archived:false"]


# --- READMEs and failures


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


# --- no silent drops


def repos_in(prompt: str) -> list[str]:
    """Return the repos a reranker prompt asks about."""
    return [line.removeprefix("### ") for line in prompt.splitlines() if line.startswith("### ")]


def test_candidates_are_split_evenly_into_batches_that_take_turns(llm: FakeLLM) -> None:
    candidates = [candidate(f"acme/{number}") for number in range(5)]

    asyncio.run(rank_candidates("Anything", llm.plan, candidates))

    assert RERANK_BATCHES == 2
    assert sorted(repos_in(prompt) for prompt in llm.ranking_prompts) == [
        ["acme/0", "acme/2", "acme/4"],
        ["acme/1", "acme/3"],
    ]
    counts = sorted("There are 3 candidates" in prompt for prompt in llm.ranking_prompts)
    assert counts == [False, True]


def test_the_batches_run_at_the_same_time(monkeypatch: pytest.MonkeyPatch) -> None:
    running = 0
    most_running = 0

    async def fake_assess(problem: str, plan: SearchPlan, batch: list[Candidate]) -> list[RepoAssessment]:
        nonlocal running, most_running
        running += 1
        most_running = max(most_running, running)
        await asyncio.sleep(0)
        running -= 1
        return [assessment(item.repo.full_name) for item in batch]

    monkeypatch.setattr("app.finder.chains._assess", fake_assess)
    plan = SearchPlan(subject="pdf", queries=["pdf"])

    results = asyncio.run(rank_candidates("Anything", plan, [candidate("acme/a"), candidate("acme/b")]))

    assert most_running == 2
    assert len(results) == 2


def test_results_from_both_batches_are_sorted_together(llm: FakeLLM) -> None:
    llm.ranking = Ranking(results=[assessment("acme/a", 40), assessment("acme/b", 95), assessment("acme/c", 70)])
    candidates = [candidate("acme/a"), candidate("acme/b"), candidate("acme/c")]

    results = asyncio.run(rank_candidates("Anything", llm.plan, candidates))

    assert [(item.full_name, item.score) for item in results] == [("acme/b", 95), ("acme/c", 70), ("acme/a", 40)]


def test_one_candidate_makes_one_call(llm: FakeLLM) -> None:
    asyncio.run(rank_candidates("Anything", llm.plan, [candidate("acme/a")]))

    assert len(llm.ranking_prompts) == 1


def test_left_out_candidates_are_marked_not_assessed_after_the_scored_ones(llm: FakeLLM) -> None:
    llm.ranking = Ranking(results=[assessment("acme/a", 20)])

    results = asyncio.run(rank_candidates("Anything", llm.plan, [candidate("acme/b"), candidate("acme/a")]))

    assert len(llm.ranking_prompts) == 2  # nothing is asked again
    assert [(item.full_name, item.score) for item in results] == [("acme/a", 20), ("acme/b", None)]
    assert results[1].why_it_fits == NOT_ASSESSED_REASON
    assert results[1].how_to_start == []


@pytest.mark.parametrize(
    "error",
    [
        ServerError(503, {"error": {"code": 503, "message": "test"}}),
        OutputParserException("not JSON"),
    ],
)
def test_a_failed_batch_only_affects_its_own_repos(
    llm: FakeLLM, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    llm.ranking = Ranking(results=[assessment("acme/a", 80), assessment("acme/b", 90)])
    llm.failing_repos["acme/b"] = error

    results = asyncio.run(rank_candidates("Anything", llm.plan, [candidate("acme/a"), candidate("acme/b")]))

    assert [(item.full_name, item.score) for item in results] == [("acme/a", 80), ("acme/b", None)]
    assert "A reranker batch failed" in caplog.text


def test_every_batch_failing_raises_the_error(llm: FakeLLM) -> None:
    error = ServerError(503, {"error": {"code": 503, "message": "test"}})
    llm.failing_repos = {"acme/a": error, "acme/b": error}

    with pytest.raises(ServerError):
        asyncio.run(rank_candidates("Anything", llm.plan, [candidate("acme/a"), candidate("acme/b")]))


def test_a_bug_in_a_batch_is_not_hidden_as_not_assessed(llm: FakeLLM) -> None:
    llm.failing_repos["acme/a"] = KeyError("a bug")

    with pytest.raises(KeyError):
        asyncio.run(rank_candidates("Anything", llm.plan, [candidate("acme/a"), candidate("acme/b")]))


def test_not_assessed_results_are_saved_and_listed_after_the_low_scores(github: FakeGitHub, llm: FakeLLM) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/low"), github_item("acme/left-out")]
    llm.ranking = Ranking(results=[assessment("acme/low", 5)])

    created = search()

    assert created.status_code == 201
    assert [(item["full_name"], item["score"]) for item in created.json()["results"]] == [
        ("acme/low", 5),
        ("acme/left-out", None),
    ]
    stored = client.get(f"/api/finder/searches/{created.json()['id']}").json()
    assert [item["score"] for item in stored["results"]] == [5, None]


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
    assert "There are 1 candidates" in system.content


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
    assert body["plan"] == {
        "subject": "pdf",
        "queries": ["pdf table", "tabular pdf"],
        "language": None,
        "must_have": ["reads tables"],
    }
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
    github.status_by_query = {"pdf table": 429, "tabular pdf": 429}  # the broader search never starts

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
            return SearchPlan(subject="fallback", queries=["from fallback"])

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


# --- indexing a result


@pytest.fixture
def result_id(github: FakeGitHub, llm: FakeLLM) -> int:
    """Run a search that finds one repo, and return that result's id."""
    github.items_by_query["pdf table"] = [github_item("acme/sample")]
    return search("Extract tables from PDFs").json()["results"][0]["id"]


def test_indexing_a_result_adds_the_repo_links_it_and_starts_indexing(
    result_id: int, cloned_urls: list[str], db_session: Session
) -> None:
    response = client.post(f"/api/finder/results/{result_id}/index")

    assert response.status_code == 202
    assert response.json()["name"] == "acme/sample"
    assert cloned_urls == ["https://github.com/acme/sample"]
    assert db_session.get(Repo, response.json()["id"]).status == "ready"
    assert db_session.get(SearchResult, result_id).repo_id == response.json()["id"]


def test_indexing_a_result_reuses_a_repo_that_was_already_added(result_id: int, cloned_urls: list[str]) -> None:
    existing = client.post("/api/repos", json={"url": "https://github.com/acme/sample"}).json()

    response = client.post(f"/api/finder/results/{result_id}/index")

    assert response.status_code == 200
    assert response.json()["id"] == existing["id"]
    assert len(cloned_urls) == 1
    assert client.get("/api/finder/searches/1").json()["results"][0]["repo_id"] == existing["id"]


def test_indexing_the_same_result_twice_does_not_index_again(result_id: int, cloned_urls: list[str]) -> None:
    first = client.post(f"/api/finder/results/{result_id}/index")
    second = client.post(f"/api/finder/results/{result_id}/index")

    assert (first.status_code, second.status_code) == (202, 200)
    assert len(cloned_urls) == 1


def test_indexing_an_unknown_result_returns_404() -> None:
    assert client.post("/api/finder/results/999/index").status_code == 404


# --- the problem guide


def guide(result: int) -> httpx.Response:
    return client.post(f"/api/finder/results/{result}/guide")


def test_a_guide_needs_a_known_result() -> None:
    assert guide(999).status_code == 404


def test_a_guide_needs_an_indexed_result(result_id: int) -> None:
    response = guide(result_id)

    assert response.status_code == 409
    assert "Index this result first" in response.json()["detail"]


def test_a_guide_needs_a_ready_repo(result_id: int, cloned_urls: list[str], db_session: Session) -> None:
    repo_id = client.post(f"/api/finder/results/{result_id}/index").json()["id"]
    db_session.get(Repo, repo_id).status = "indexing"
    db_session.commit()

    assert guide(result_id).status_code == 409


def test_a_guide_needs_the_embedding_model_in_use(result_id: int, cloned_urls: list[str], db_session: Session) -> None:
    repo_id = client.post(f"/api/finder/results/{result_id}/index").json()["id"]
    db_session.get(Repo, repo_id).embedding_model = "other"
    db_session.commit()

    assert guide(result_id).json()["detail"].startswith("Re-index needed")


def test_the_guide_is_saved_and_returned_with_the_problem_and_repo(
    result_id: int, cloned_urls: list[str], db_session: Session
) -> None:
    repo_id = client.post(f"/api/finder/results/{result_id}/index").json()["id"]

    response = guide(result_id)

    assert response.status_code == 200
    body = response.json()
    assert body["result_id"] == result_id
    assert body["problem"] == "Extract tables from PDFs"
    assert body["repo_id"] == repo_id
    assert body["repo_name"] == "acme/sample"
    assert body["guide"].endswith(FAKE_CHAT_RESPONSES[0])
    assert db_session.get(SearchResult, result_id).problem_guide == body["guide"]
    assert client.get("/api/finder/searches/1").json()["results"][0]["problem_guide"] == body["guide"]


def test_the_guide_starts_with_the_problem_word_for_word(
    github: FakeGitHub, llm: FakeLLM, cloned_urls: list[str]
) -> None:
    github.items_by_query["pdf table"] = [github_item("acme/sample")]
    problem = "Extract tables from PDFs,\nthen {load} them into my   database!"
    result = search(problem).json()["results"][0]["id"]
    client.post(f"/api/finder/results/{result}/index")

    text = guide(result).json()["guide"]

    assert text.startswith(
        "## Your problem\n\n> Extract tables from PDFs,\n> then {load} them into my   database!\n\n"
    )


def test_the_guide_prompt_has_every_section_and_the_material(
    result_id: int, cloned_urls: list[str], recording_model: RecordingChatModel
) -> None:
    client.post(f"/api/finder/results/{result_id}/index")

    guide(result_id)

    system, human = recording_model.prompts[0]
    for section in [
        "## Recommended repo",
        "## Does it solve your problem?",
        "## Setup",
        "## Step-by-step: solving YOUR problem with this repo",
        "## What you still need to build yourself",
        "## Gotchas and next steps",
    ]:
        assert section in system.content
    assert "**Fully**, **Partly** or **No**" in system.content
    assert "https://github.com/acme/sample" in system.content
    assert "Only cite a file if it appears" in system.content
    assert "untrusted" in system.content
    assert "src/\n  app.py" in system.content  # file tree
    assert "A tiny project used by the RepoScout tests." in system.content  # README
    assert "def greet" in system.content  # excerpts
    assert "About acme/sample" in system.content  # GitHub description
    assert "Extract tables from PDFs" in human.content
    assert "acme/sample" in human.content


class StubStore:
    """A vector store that records searches and returns the given chunks, in order, for every search."""

    def __init__(self, chunks: list[tuple[str, str]]) -> None:
        self.chunks = chunks
        self.searches: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.searches.append((query, k))
        return [
            Document(f"text of {path}", metadata={"path": path, "start_line": 1, "end_line": 2, "language": language})
            for path, language in self.chunks[:k]
        ]


def test_the_guide_searches_for_code_then_the_problem_and_setup_topics(
    result_id: int, cloned_urls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post(f"/api/finder/results/{result_id}/index")
    store = StubStore([("a.py", "Python")])
    monkeypatch.setattr("app.rag.get_vectorstore", lambda repo: store)

    assert guide(result_id).status_code == 200
    assert store.searches == [
        *((probe, PROBLEM_GUIDE_CODE_PROBE_K * FILTERED_FETCH_MULTIPLIER) for probe in PROBLEM_GUIDE_CODE_PROBES),
        ("Extract tables from PDFs", PROBLEM_EXCERPT_K * FILTERED_FETCH_MULTIPLIER),
        *((probe, PROBLEM_GUIDE_PROBE_K * FILTERED_FETCH_MULTIPLIER) for probe in PROBLEM_GUIDE_PROBES),
    ]


def test_the_guide_gets_code_first_and_never_built_docs(
    result_id: int, cloned_urls: list[str], recording_model: RecordingChatModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post(f"/api/finder/results/{result_id}/index")
    store = StubStore(
        [
            ("docs/_build/html/code.html", "HTML"),
            ("docs/index.html", "HTML"),
            ("docs/usage.md", "Markdown"),
            ("src/loader.py", "Python"),
        ]
    )
    monkeypatch.setattr("app.rag.get_vectorstore", lambda repo: store)

    guide(result_id)

    excerpts = recording_model.prompts[0][0].content.split("Excerpts:")[-1]
    assert "docs/_build" not in excerpts
    assert "docs/index.html" not in excerpts
    assert excerpts.index("src/loader.py") < excerpts.index("docs/usage.md")


def test_a_filter_returns_up_to_k_of_the_chunks_that_pass_it(monkeypatch: pytest.MonkeyPatch) -> None:
    store = StubStore([("skip.html", "HTML"), ("a.py", "Python"), ("b.py", "Python"), ("c.py", "Python")])
    monkeypatch.setattr("app.rag.get_vectorstore", lambda repo: store)

    documents = collect_excerpts(1, [("query", 2)], keep=lambda item: item.metadata["language"] == "Python")

    assert [item.metadata["path"] for item in documents] == ["a.py", "b.py"]
    assert store.searches == [("query", 2 * FILTERED_FETCH_MULTIPLIER)]


def test_the_prompt_bases_the_verdict_on_code(
    result_id: int, cloned_urls: list[str], recording_model: RecordingChatModel
) -> None:
    client.post(f"/api/finder/results/{result_id}/index")

    guide(result_id)

    system = recording_model.prompts[0][0].content
    assert "Base this on the code, not on what the README promises" in system
    assert "When unsure, say **Partly**" in system
    assert "the code is right" in system


def test_a_new_guide_replaces_the_old_one(
    result_id: int, cloned_urls: list[str], recording_model: RecordingChatModel
) -> None:
    client.post(f"/api/finder/results/{result_id}/index")
    guide(result_id)
    recording_model.responses = ["A newer guide."]
    recording_model.i = 0

    assert guide(result_id).json()["guide"].endswith("A newer guide.")


def test_a_failed_guide_returns_503_and_keeps_the_old_one(
    result_id: int, cloned_urls: list[str], db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post(f"/api/finder/results/{result_id}/index")
    old = guide(result_id).json()["guide"]

    def fail(*_: Any) -> str:
        raise ServerError(503, {"error": {"code": 503, "message": "test"}})

    monkeypatch.setattr("app.routers.finder.generate_problem_guide", fail)

    response = guide(result_id)

    assert response.status_code == 503
    assert "busy" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.get(SearchResult, result_id).problem_guide == old


def test_a_missing_setting_returns_503_for_the_guide(
    result_id: int, cloned_urls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post(f"/api/finder/results/{result_id}/index")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GUIDE_MODELS", "guide-model-name")
    get_settings.cache_clear()

    response = guide(result_id)

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_deleting_the_repo_means_indexing_again(result_id: int, cloned_urls: list[str]) -> None:
    repo_id = client.post(f"/api/finder/results/{result_id}/index").json()["id"]
    assert client.delete(f"/api/repos/{repo_id}").status_code == 204

    assert guide(result_id).status_code == 409


# --- verifying the guide against the code

DRAFT = (
    "## Recommended repo\n[acme/sample](https://github.com/acme/sample) greets people.\n\n"
    "## Does it solve your problem?\n**{verdict}**. It reads PDFs, see `src/app.py`.\n\n"
    "## Setup\n1. Install it.\n\n"
    "## Step-by-step: solving YOUR problem with this repo\n1. Run `src/app.py`.\n\n"
    "## What you still need to build yourself\nNothing.\n\n"
    "## Gotchas and next steps\nNone."
)
FULLY_DRAFT = DRAFT.format(verdict="Fully")
PARTLY_DRAFT = DRAFT.format(verdict="Partly")


class FakeVerifier:
    """Replaces the guide's structured model: replies as scripted and records what it was sent."""

    def __init__(self) -> None:
        self.reply: VerifiedGuide | Exception = VerifiedGuide(guide=FULLY_DRAFT, changes=[])
        self.prompts: list[list[BaseMessage]] = []

    def get_model(self, schema: type, fake_reply: Callable[[Any], Any] | None = None) -> Runnable:
        assert schema is VerifiedGuide

        def respond(prompt: PromptValue) -> VerifiedGuide:
            self.prompts.append(prompt.to_messages())
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

        return RunnableLambda(respond)


@pytest.fixture
def verifier(
    result_id: int,
    cloned_urls: list[str],
    recording_model: RecordingChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> FakeVerifier:
    """Index the result, script the draft and replace the verify call."""
    client.post(f"/api/finder/results/{result_id}/index")
    recording_model.responses = [FULLY_DRAFT]
    recording_model.i = 0
    fake = FakeVerifier()
    monkeypatch.setattr("app.finder.chains.get_structured_guide_model", fake.get_model)
    return fake


def test_the_verifier_gets_the_problem_the_draft_and_the_cited_code(result_id: int, verifier: FakeVerifier) -> None:
    guide(result_id)

    system, human = verifier.prompts[0]
    assert "### src/app.py" in system.content
    assert "def greet" in system.content
    assert "untrusted" in system.content
    assert "**Fully** is only right if the code" in system.content
    assert "src/\n  app.py" in system.content  # file tree
    assert "A tiny project used by the RepoScout tests." in system.content  # README
    assert "never remove one just because the code doesn't confirm it" in system.content
    assert "Only remove what the code contradicts" in system.content
    assert "A file listed in the file tree exists" in system.content
    assert "Extract tables from PDFs" in human.content
    assert FULLY_DRAFT in human.content


def test_a_corrected_guide_is_saved_with_a_record_of_the_change(
    result_id: int, verifier: FakeVerifier, db_session: Session
) -> None:
    corrected = PARTLY_DRAFT.replace("It reads PDFs", "It only reads images")
    verifier.reply = VerifiedGuide(guide=corrected, changes=["The code only accepts images."])

    body = guide(result_id).json()

    assert body["verdict"] == "Partly"
    assert body["verification"] == {
        "status": "verified",
        "draft": FULLY_DRAFT,
        "changed": True,
        "changes": ["The code only accepts images."],
        "files_checked": ["src/app.py"],
    }
    assert body["guide"].startswith("## Your problem\n\n> Extract tables from PDFs\n\n## Recommended repo")
    assert "It only reads images" in body["guide"]
    assert "It reads PDFs" not in body["guide"]
    assert db_session.get(SearchResult, result_id).problem_guide == body["guide"]


def test_the_draft_and_verification_are_saved_next_to_the_guide(
    result_id: int, verifier: FakeVerifier, db_session: Session
) -> None:
    verifier.reply = VerifiedGuide(guide=PARTLY_DRAFT, changes=["The code only accepts images."])

    body = guide(result_id).json()

    saved = db_session.get(SearchResult, result_id).guide_verification
    assert saved == body["verification"]
    assert saved["draft"] == FULLY_DRAFT
    stored = client.get("/api/finder/searches/1").json()["results"][0]
    assert stored["guide_verification"] == body["verification"]


def test_a_failed_verification_also_saves_the_draft(result_id: int, verifier: FakeVerifier, db_session: Session) -> None:
    verifier.reply = ServerError(503, {"error": {"code": 503, "message": "test"}})

    guide(result_id)

    saved = db_session.get(SearchResult, result_id).guide_verification
    assert saved["status"] == "failed"
    assert saved["draft"] == FULLY_DRAFT


def test_a_result_without_a_guide_has_no_verification(result_id: int) -> None:
    assert client.get("/api/finder/searches/1").json()["results"][0]["guide_verification"] is None


def test_a_draft_that_was_right_is_recorded_as_unchanged(result_id: int, verifier: FakeVerifier) -> None:
    body = guide(result_id).json()

    assert body["verdict"] == "Fully"
    assert body["verification"]["status"] == "verified"
    assert body["verification"]["changed"] is False
    assert body["verification"]["changes"] == []
    assert "Not verified" not in body["guide"]


@pytest.mark.parametrize(
    "reply",
    [
        ServerError(503, {"error": {"code": 503, "message": "test"}}),
        OutputParserException("not JSON"),
        VerifiedGuide(guide="   ", changes=[]),
        VerifiedGuide(guide="## Recommended repo\nThe rest went missing.", changes=[]),
    ],
)
def test_a_failed_verification_keeps_the_draft_labelled_and_lowers_fully(
    result_id: int, verifier: FakeVerifier, reply: VerifiedGuide | Exception, db_session: Session
) -> None:
    verifier.reply = reply

    response = guide(result_id)

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "Partly"
    assert body["verification"]["status"] == "failed"
    assert body["verification"]["changed"] is True
    assert body["verification"]["files_checked"] == ["src/app.py"]
    assert "Lowered the verdict" in body["verification"]["changes"][0]
    assert "> **Not verified against the code.**" in body["guide"]
    assert "**Fully**" not in body["guide"]
    assert "It reads PDFs" in body["guide"]
    assert db_session.get(SearchResult, result_id).problem_guide == body["guide"]


def test_a_failed_verification_leaves_a_lower_verdict_alone(
    result_id: int, verifier: FakeVerifier, recording_model: RecordingChatModel
) -> None:
    recording_model.responses = [DRAFT.format(verdict="No")]
    recording_model.i = 0
    verifier.reply = ServerError(503, {"error": {"code": 503, "message": "test"}})

    body = guide(result_id).json()

    assert body["verdict"] == "No"
    assert body["verification"]["changed"] is False
    assert body["verification"]["changes"] == []


def test_no_source_files_skips_the_check_and_labels_the_guide(
    result_id: int, verifier: FakeVerifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.finder.chains.load_guide_sources", lambda *_: [])

    body = guide(result_id).json()

    assert verifier.prompts == []
    assert body["verification"]["status"] == "skipped"
    assert body["verdict"] == "Partly"
    assert "names no source files" in body["guide"]


def test_a_draft_error_still_fails_the_request(result_id: int, verifier: FakeVerifier, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: str) -> Any:
        raise ServerError(503, {"error": {"code": 503, "message": "test"}})

    monkeypatch.setattr("app.finder.chains.get_guide_model", lambda: RunnableLambda(fail))

    assert guide(result_id).status_code == 503
    assert verifier.prompts == []


def test_the_fake_provider_verifies_offline(result_id: int, cloned_urls: list[str]) -> None:
    client.post(f"/api/finder/results/{result_id}/index")

    body = guide(result_id).json()

    assert body["verification"]["status"] == "verified"
    assert body["verification"]["changed"] is False
    assert body["verification"]["files_checked"] == ["src/app.py"]
    assert body["verdict"] is None


def test_the_verifier_uses_the_guide_chain_at_the_guide_thinking_level(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    levels: list[str] = []
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder-key")
    monkeypatch.setenv("CHAT_MODELS", "gemini-3-chat")
    monkeypatch.setenv("GUIDE_MODELS", "gemini-3-guide")
    monkeypatch.setenv("CHAT_THINKING_LEVEL", "minimal")
    monkeypatch.setenv("GUIDE_THINKING_LEVEL", "low")

    def build(name: str, is_last: bool, thinking_level: str) -> StructuredFake:
        levels.append(thinking_level)
        return StructuredFake(responses=["unused"], label=name, calls=calls, failing={})

    monkeypatch.setattr("app.llm._build_chat_model", build)

    get_structured_guide_model(VerifiedGuide).invoke("Hi")

    assert calls == ["gemini-3-guide"]
    assert levels == ["low"]
