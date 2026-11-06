"""Endpoints to read and generate a repo's onboarding guide."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import CurrentUser, current_user, get_repo_or_404
from app.db import get_db
from app.demo import limit
from app.llm import GEMINI_ERRORS, describe_gemini_error
from app.rag import generate_onboarding_guide
from app.routers.chat import require_searchable
from app.routers.repos import ensure_clone_or_503
from app.schemas import GuideOut

router = APIRouter(prefix="/api/repos/{repo_id}/guide", tags=["guide"])


@router.get("", response_model=GuideOut)
def get_guide(repo_id: int, user: CurrentUser | None = Depends(current_user), db: Session = Depends(get_db)) -> GuideOut:
    """Return the saved guide. It's written from the repo's code alone, so everyone who can see the repo shares it."""
    repo = get_repo_or_404(db, repo_id, user)
    if repo.onboarding_guide is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No guide yet. POST to this URL to generate one.")
    return GuideOut(repo_id=repo.id, guide=repo.onboarding_guide)


@router.post("", response_model=GuideOut, dependencies=[Depends(limit("guide"))])
def create_guide(repo_id: int, user: CurrentUser | None = Depends(current_user), db: Session = Depends(get_db)) -> GuideOut:
    """Generate the guide and save it, replacing any earlier one. This can take a while."""
    repo = get_repo_or_404(db, repo_id, user)
    require_searchable(repo)
    ensure_clone_or_503(repo)
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
