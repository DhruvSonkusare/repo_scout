"""Endpoints to find repos for a problem and read past searches."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Search, SearchResult, SessionLocal, get_db
from app.finder.chains import Candidate, RepoAssessment, SearchPlan, find_repos, rank_candidates
from app.finder.github import GitHubError, GitHubRateLimitError
from app.llm import GEMINI_ERRORS, ModelSetupError, describe_gemini_error
from app.schemas import SearchOut, SearchRequest, SearchSummaryOut

router = APIRouter(prefix="/api/finder", tags=["finder"])

UNREADABLE_REPLY_MESSAGE = "The model's reply couldn't be read. Try the search again."


def _save_search(
    problem: str, plan: SearchPlan, candidates: list[Candidate], assessments: list[RepoAssessment]
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
