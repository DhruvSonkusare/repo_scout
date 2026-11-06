"""Endpoints to add, inspect, re-index and delete repos, and to read their files."""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, Response, status
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import CurrentUser, current_user, get_repo_or_404, in_workspace, visible_repos
from app.config import get_settings
from app.db import IN_PROGRESS_STATUSES, Conversation, Repo, SearchResult, UserRepo, get_db
from app.demo import check_index_rate, check_repo_size, count_user_index, require_index_password
from app.ingest import CloneError, ensure_clone, index_repo, parse_github_url, remove_clone
from app.schemas import RepoCreate, RepoFileOut, RepoOut
from app.vectorstore import delete_collection

router = APIRouter(prefix="/api/repos", tags=["repos"])


def _require_idle(repo: Repo) -> None:
    """Raise a 409 if a background job is using this repo's clone and collection."""
    if repo.status in IN_PROGRESS_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Repo is busy ({repo.status}). Try again when it finishes.")


def ensure_clone_or_503(repo: Repo) -> Path:
    """Return a repo's clone, restoring it if needed, or raise a 503 if GitHub can't provide it."""
    try:
        return ensure_clone(repo)
    except CloneError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error


def _link_to_workspace(db: Session, repo_id: int, user: CurrentUser) -> None:
    """Record that a user added a repo. Adding it twice changes nothing."""
    db.execute(insert(UserRepo).values(user_id=user.id, repo_id=repo_id).on_conflict_do_nothing())
    db.commit()


def add_or_get_repo(
    db: Session, github_url: str, request: Request, password: str, user: CurrentUser | None
) -> tuple[Repo, bool]:
    """Return the repo for a GitHub URL, adding it if needed, and whether it was just added.

    Repos are shared and indexed once, so a repo that's already here costs nothing to add. A new one
    takes the demo's index password from a guest and becomes public, or uses up a signed-in user's
    limits and stays in their workspace. A repo only users have added takes the password from a guest
    too, so guests can't tell it's here, and it becomes public.
    """
    try:
        name = parse_github_url(github_url.strip())
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    url = f"https://github.com/{name}"

    repo = db.scalar(select(Repo).where(Repo.url == url))
    if repo is not None:
        if user:
            _link_to_workspace(db, repo.id, user)
        elif not repo.is_public:
            require_index_password(request, password)
            repo.is_public = True
            db.commit()
        return repo, False

    if user:
        check_index_rate(request, user)
    else:
        require_index_password(request, password)
    check_repo_size(name)
    if user:
        count_user_index(user)
    repo = Repo(url=url, name=name, is_public=user is None)
    db.add(repo)
    created = True
    try:
        db.commit()
    except IntegrityError:
        # Another request added the same URL between the check above and this commit.
        db.rollback()
        repo = db.scalars(select(Repo).where(Repo.url == url)).one()
        created = False
    if user:
        _link_to_workspace(db, repo.id, user)
    return repo, created


@router.post("", response_model=RepoOut, status_code=status.HTTP_201_CREATED)
def add_repo(
    body: RepoCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
    x_index_password: str = Header(default=""),
    user: CurrentUser | None = Depends(current_user),
    db: Session = Depends(get_db),
) -> Repo:
    """Add a repo and start indexing it. A repo that was already added is returned as it is."""
    repo, created = add_or_get_repo(db, body.url, request, x_index_password, user)
    if created:
        background_tasks.add_task(index_repo, repo.id, user is not None)
    else:
        response.status_code = status.HTTP_200_OK
    return repo


@router.get("", response_model=list[RepoOut])
def list_repos(user: CurrentUser | None = Depends(current_user), db: Session = Depends(get_db)) -> list[Repo]:
    """List the repos this user can see, newest first."""
    return list(db.scalars(select(Repo).where(visible_repos(user)).order_by(Repo.created_at.desc(), Repo.id.desc())))


@router.get("/{repo_id}", response_model=RepoOut)
def get_repo(repo_id: int, user: CurrentUser | None = Depends(current_user), db: Session = Depends(get_db)) -> Repo:
    """Return one repo. Poll this to follow indexing progress."""
    return get_repo_or_404(db, repo_id, user)


@router.post("/{repo_id}/reindex", response_model=RepoOut, status_code=status.HTTP_202_ACCEPTED)
def reindex_repo(
    repo_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    x_index_password: str = Header(default=""),
    user: CurrentUser | None = Depends(current_user),
    db: Session = Depends(get_db),
) -> Repo:
    """Clone and index a repo again from scratch. Public repos take the index password, even when signed in."""
    if user is None:
        require_index_password(request, x_index_password)
    repo = get_repo_or_404(db, repo_id, user)
    own_repo = not repo.is_public and in_workspace(db, repo_id, user)
    if user and not own_repo:
        require_index_password(request, x_index_password)
    _require_idle(repo)
    if own_repo:
        check_index_rate(request, user)
        count_user_index(user)
    repo.status = "pending"
    repo.error = None
    repo.chunks_embedded = 0
    db.commit()
    background_tasks.add_task(index_repo, repo.id, own_repo)
    return repo


def _remove_from_workspace(db: Session, repo_id: int, user: CurrentUser) -> None:
    """Take a repo out of a user's workspace, with their chats about it. Their saved guides stay."""
    db.execute(delete(Conversation).where(Conversation.repo_id == repo_id, Conversation.owner_user_id == user.id))
    db.execute(
        update(SearchResult)
        .where(SearchResult.repo_id == repo_id, SearchResult.owner_user_id == user.id)
        .values(repo_id=None)
    )
    db.execute(delete(UserRepo).where(UserRepo.repo_id == repo_id, UserRepo.user_id == user.id))


@router.delete("/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_repo(
    repo_id: int,
    request: Request,
    x_index_password: str = Header(default=""),
    user: CurrentUser | None = Depends(current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete a repo with its clone, vectors and chats.

    Repos are shared, so a user who added one only removes it from their workspace. The repo itself
    goes when it isn't public and nobody else has it. Other cases take the index password.
    """
    if user is None:
        require_index_password(request, x_index_password)
    repo = get_repo_or_404(db, repo_id, user)
    own_link = in_workspace(db, repo_id, user)
    if user and not own_link:
        require_index_password(request, x_index_password)
    _require_idle(repo)
    if own_link:
        _remove_from_workspace(db, repo_id, user)
        if repo.is_public or db.scalar(select(UserRepo.user_id).where(UserRepo.repo_id == repo_id).limit(1)):
            db.commit()
            return
    # Remove the files first: if that fails, the row remains and the delete can be retried.
    delete_collection(repo_id)
    remove_clone(repo_id)
    db.delete(repo)
    db.commit()


@router.get("/{repo_id}/file", response_model=RepoFileOut)
def read_repo_file(
    repo_id: int, path: str, user: CurrentUser | None = Depends(current_user), db: Session = Depends(get_db)
) -> RepoFileOut:
    """Return a file from a repo's clone, for the citation viewer."""
    repo = get_repo_or_404(db, repo_id, user)
    settings = get_settings()
    not_found = HTTPException(status.HTTP_404_NOT_FOUND, "File not found.")
    if repo.status == "ready":
        ensure_clone_or_503(repo)
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
