"""Endpoints to find repos for a problem, read past searches and get a guide to using a result."""

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Repo, Search, SearchResult, SessionLocal, get_db
from app.finder.chains import (
    Candidate,
    RepoResult,
    SearchPlan,
    find_repos,
    generate_problem_guide,
    rank_candidates,
)
from app.finder.github import GitHubError, GitHubRateLimitError
from app.ingest import index_repo
from app.llm import GEMINI_ERRORS, ModelSetupError, describe_gemini_error
from app.routers.chat import require_searchable
from app.routers.repos import add_or_get_repo
from app.schemas import GuideVerificationOut, ProblemGuideOut, RepoOut, SearchOut, SearchRequest, SearchSummaryOut

router = APIRouter(prefix="/api/finder", tags=["finder"])

UNREADABLE_REPLY_MESSAGE = "The model's reply couldn't be read. Try the search again."


def _save_search(
    problem: str, plan: SearchPlan, candidates: list[Candidate], assessments: list[RepoResult]
) -> SearchOut:
    """Save the search and its results, and return them as stored."""
    repos = {candidate.repo.full_name: candidate.repo for candidate in candidates}
    with SessionLocal() as db:
        search = Search(problem=problem, plan=plan.model_dump())
        search.results = [
            SearchResult(
                full_name=assessment.full_name,
                url=repos[assessment.full_name].html_url,
                description=repos[assessment.full_name].description,
                stars=repos[assessment.full_name].stars,
                language=repos[assessment.full_name].language,
                last_pushed=repos[assessment.full_name].pushed_at,
                score=assessment.score,
                why_it_fits=assessment.why_it_fits,
                caveats=assessment.caveats,
                how_to_start=assessment.how_to_start,
            )
            for assessment in assessments
        ]
        db.add(search)
        db.commit()
        return SearchOut.model_validate(search)


@router.post("/search", response_model=SearchOut, status_code=status.HTTP_201_CREATED)
async def search_repos(body: SearchRequest) -> SearchOut:
    """Find and rank repos for a problem. Takes a while, because it makes two LLM calls and many GitHub requests."""
    try:
        plan, candidates = await find_repos(body.problem)
        assessments = await rank_candidates(body.problem, plan, candidates)
    except GitHubRateLimitError as error:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(error)) from error
    except GitHubError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
    except GEMINI_ERRORS as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, describe_gemini_error(error)) from error
    except ModelSetupError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    except (OutputParserException, ValidationError) as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, UNREADABLE_REPLY_MESSAGE) from error
    return await asyncio.to_thread(_save_search, body.problem, plan, candidates, assessments)


@router.get("/searches", response_model=list[SearchSummaryOut])
def list_searches(db: Session = Depends(get_db)) -> list[Search]:
    """List past searches, newest first."""
    return list(db.scalars(select(Search).order_by(Search.id.desc())))


@router.get("/searches/{search_id}", response_model=SearchOut)
def get_search(search_id: int, db: Session = Depends(get_db)) -> Search:
    """Return one past search with its results, best match first."""
    search = db.get(Search, search_id)
    if search is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Search not found.")
    return search


def _get_result_or_404(db: Session, result_id: int) -> SearchResult:
    """Return a search result, or raise a 404."""
    result = db.get(SearchResult, result_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Search result not found.")
    return result


@router.post("/results/{result_id}/index", response_model=RepoOut, status_code=status.HTTP_202_ACCEPTED)
def index_result(
    result_id: int, background_tasks: BackgroundTasks, response: Response, db: Session = Depends(get_db)
) -> Repo:
    """Link a result to its repo and start indexing it. A repo that was already added is reused as it is."""
    result = _get_result_or_404(db, result_id)
    repo, created = add_or_get_repo(db, result.url)
    result.repo_id = repo.id
    db.commit()
    if created:
        background_tasks.add_task(index_repo, repo.id)
    else:
        response.status_code = status.HTTP_200_OK
    return repo


@router.post("/results/{result_id}/guide", response_model=ProblemGuideOut)
def create_problem_guide(result_id: int, db: Session = Depends(get_db)) -> ProblemGuideOut:
    """Write a guide, check it against the repo's code, and save it, replacing any earlier one. This can take a while."""
    result = _get_result_or_404(db, result_id)
    repo = result.repo
    if repo is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Index this result first: POST to its /index URL.")
    require_searchable(repo)
    try:
        guide = generate_problem_guide(result.search.problem, repo, result.description)
    except GEMINI_ERRORS as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, describe_gemini_error(error)) from error
    except ValueError as error:
        # The model factory raises this when a setting is missing from .env.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    result.problem_guide = guide.guide
    db.commit()
    return ProblemGuideOut(
        result_id=result.id,
        problem=result.search.problem,
        repo_id=repo.id,
        repo_name=repo.name,
        guide=guide.guide,
        verdict=guide.verdict,
        verification=GuideVerificationOut.model_validate(guide.verification),
    )
