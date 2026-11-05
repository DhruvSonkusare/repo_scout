"""Endpoints to read and generate a repo's onboarding guide."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import Repo, get_db
from app.llm import GEMINI_ERRORS, describe_gemini_error
from app.rag import generate_onboarding_guide
from app.routers.chat import require_searchable
from app.schemas import GuideOut

router = APIRouter(prefix="/api/repos/{repo_id}/guide", tags=["guide"])


def _get_repo_or_404(db: Session, repo_id: int) -> Repo:
    """Return a repo, or raise a 404."""
    repo = db.get(Repo, repo_id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not found.")
    return repo


@router.get("", response_model=GuideOut)
def get_guide(repo_id: int, db: Session = Depends(get_db)) -> GuideOut:
    """Return the saved guide."""
    repo = _get_repo_or_404(db, repo_id)
    if repo.onboarding_guide is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No guide yet. POST to this URL to generate one.")
    return GuideOut(repo_id=repo.id, guide=repo.onboarding_guide)


@router.post("", response_model=GuideOut)
def create_guide(repo_id: int, db: Session = Depends(get_db)) -> GuideOut:
    """Generate the guide and save it, replacing any earlier one. This can take a while."""
    repo = _get_repo_or_404(db, repo_id)
    require_searchable(repo)
    try:
        guide = generate_onboarding_guide(repo)
    except GEMINI_ERRORS as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, describe_gemini_error(error)) from error
    except ValueError as error:
        # The model factory raises this when a setting is missing from .env.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    repo.onboarding_guide = guide
    db.commit()
    return GuideOut(repo_id=repo.id, guide=guide)
