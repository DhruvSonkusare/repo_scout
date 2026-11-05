"""Endpoints to add, inspect, re-index and delete repos, and to read their files."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import IN_PROGRESS_STATUSES, Repo, get_db
from app.ingest import index_repo, parse_github_url, remove_clone
from app.schemas import RepoCreate, RepoFileOut, RepoOut
from app.vectorstore import delete_collection

router = APIRouter(prefix="/api/repos", tags=["repos"])


def _get_repo_or_404(db: Session, repo_id: int) -> Repo:
    """Return a repo, or raise a 404."""
    repo = db.get(Repo, repo_id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not found.")
    return repo


def _require_idle(repo: Repo) -> None:
    """Raise a 409 if a background job is using this repo's clone and collection."""
    if repo.status in IN_PROGRESS_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Repo is busy ({repo.status}). Try again when it finishes.")


@router.post("", response_model=RepoOut, status_code=status.HTTP_201_CREATED)
def add_repo(
    body: RepoCreate, background_tasks: BackgroundTasks, response: Response, db: Session = Depends(get_db)
) -> Repo:
    """Add a repo and start indexing it. A repo that was already added is returned as it is."""
    try:
        name = parse_github_url(body.url.strip())
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    url = f"https://github.com/{name}"

    existing = db.scalar(select(Repo).where(Repo.url == url))
    if existing is not None:
        response.status_code = status.HTTP_200_OK
        return existing

    repo = Repo(url=url, name=name)
    db.add(repo)
    try:
        db.commit()
    except IntegrityError:
        # Another request added the same URL between the check above and this commit.
        db.rollback()
        response.status_code = status.HTTP_200_OK
        return db.scalars(select(Repo).where(Repo.url == url)).one()
    background_tasks.add_task(index_repo, repo.id)
    return repo


@router.get("", response_model=list[RepoOut])
def list_repos(db: Session = Depends(get_db)) -> list[Repo]:
    """List all repos, newest first."""
    return list(db.scalars(select(Repo).order_by(Repo.created_at.desc(), Repo.id.desc())))


@router.get("/{repo_id}", response_model=RepoOut)
def get_repo(repo_id: int, db: Session = Depends(get_db)) -> Repo:
    """Return one repo. Poll this to follow indexing progress."""
    return _get_repo_or_404(db, repo_id)


@router.post("/{repo_id}/reindex", response_model=RepoOut, status_code=status.HTTP_202_ACCEPTED)
def reindex_repo(repo_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> Repo:
    """Clone and index a repo again from scratch."""
    repo = _get_repo_or_404(db, repo_id)
    _require_idle(repo)
    repo.status = "pending"
    repo.error = None
    repo.chunks_embedded = 0
    db.commit()
    background_tasks.add_task(index_repo, repo.id)
    return repo


@router.delete("/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_repo(repo_id: int, db: Session = Depends(get_db)) -> None:
    """Delete a repo with its clone, vectors and chats."""
    repo = _get_repo_or_404(db, repo_id)
    _require_idle(repo)
    # Remove the files first: if that fails, the row remains and the delete can be retried.
    delete_collection(repo_id)
    remove_clone(repo_id)
    db.delete(repo)
    db.commit()


@router.get("/{repo_id}/file", response_model=RepoFileOut)
def read_repo_file(repo_id: int, path: str, db: Session = Depends(get_db)) -> RepoFileOut:
    """Return a file from a repo's clone, for the citation viewer."""
    _get_repo_or_404(db, repo_id)
    settings = get_settings()
    not_found = HTTPException(status.HTTP_404_NOT_FOUND, "File not found.")
    clone = (settings.repos_dir / str(repo_id)).resolve()
    try:
        # resolve() follows `..` and symlinks, so this checks where the path really leads.
        target = (clone / path).resolve()
        relative = target.relative_to(clone)
        if ".git" in relative.parts or not target.is_file():
            raise not_found
        size = target.stat().st_size
    except (OSError, ValueError) as error:
        # Raised for paths outside the clone, and for invalid paths such as ones with a null byte.
        raise not_found from error

    if size > settings.max_file_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "File is too large to display.")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File is not UTF-8 text.") from error
    return RepoFileOut(path=relative.as_posix(), content=content)
