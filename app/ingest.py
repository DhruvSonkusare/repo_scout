"""Turn a GitHub repo into chunks ready for indexing: clone, filter files, split them."""

import logging
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import IN_PROGRESS_STATUSES, Repo, SessionLocal
from app.llm import (
    GEMINI_ERRORS,
    describe_gemini_error,
    embedding_identity,
    gemini_retry_delay,
    is_daily_quota_error,
)
from app.vectorstore import delete_collection, get_vectorstore

logger = logging.getLogger(__name__)

CLONE_TIMEOUT_SECONDS = 300
# Chunks between progress updates for embeddings that run on this machine.
LOCAL_EMBED_BATCH_SIZE = 32
MAX_TREE_DEPTH = 3
MAX_TREE_LINES = 300

GITHUB_URL_PATTERN = re.compile(r"https://github\.com/([A-Za-z0-9_-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?")
INVALID_REPO_NAMES = {".", "..", ".git"}
# Cloning runs unattended, so a private or missing repo must fail, not wait for a login: the
# empty helper stops Windows login pop-ups, and GIT_TERMINAL_PROMPT stops terminal prompts.
GIT_NO_LOGIN = ["-c", "credential.helper="]
_restore_lock = threading.Lock()

SKIPPED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "bower_components", "venv", ".venv", "__pycache__",
    "site-packages", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
    "dist", "build", "target", "out", "vendor", ".next", ".nuxt", "coverage", "htmlcov",
}  # fmt: skip
LOCKFILES = {
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "bun.lock",
    "poetry.lock", "Pipfile.lock", "uv.lock", "Cargo.lock", "Gemfile.lock", "composer.lock", "go.sum",
}  # fmt: skip
MINIFIED_SUFFIXES = (".min.js", ".min.css", ".bundle.js")

# Suffix -> (language name for stats, splitter rules). None means the file type has no
# LangChain rules, so it gets the generic splitter (paragraphs, then lines, then words).
LANGUAGE_BY_SUFFIX: dict[str, tuple[str, Language | None]] = {
    ".py": ("Python", Language.PYTHON),
    ".js": ("JavaScript", Language.JS),
    ".jsx": ("JavaScript", Language.JS),
    ".mjs": ("JavaScript", Language.JS),
    ".cjs": ("JavaScript", Language.JS),
    ".ts": ("TypeScript", Language.TS),
    ".tsx": ("TypeScript", Language.TS),
    ".go": ("Go", Language.GO),
    ".rs": ("Rust", Language.RUST),
    ".java": ("Java", Language.JAVA),
    ".kt": ("Kotlin", Language.KOTLIN),
    ".kts": ("Kotlin", Language.KOTLIN),
    ".c": ("C", Language.C),
    ".h": ("C", Language.C),
    ".cpp": ("C++", Language.CPP),
    ".cc": ("C++", Language.CPP),
    ".cxx": ("C++", Language.CPP),
    ".hpp": ("C++", Language.CPP),
    ".cs": ("C#", Language.CSHARP),
    ".rb": ("Ruby", Language.RUBY),
    ".php": ("PHP", Language.PHP),
    ".swift": ("Swift", Language.SWIFT),
    ".scala": ("Scala", Language.SCALA),
    ".lua": ("Lua", Language.LUA),
    ".hs": ("Haskell", Language.HASKELL),
    ".ex": ("Elixir", Language.ELIXIR),
    ".exs": ("Elixir", Language.ELIXIR),
    ".r": ("R", Language.R),
    ".ps1": ("PowerShell", Language.POWERSHELL),
    ".sol": ("Solidity", Language.SOL),
    ".proto": ("Protobuf", Language.PROTO),
    ".md": ("Markdown", Language.MARKDOWN),
    ".mdx": ("Markdown", Language.MARKDOWN),
    ".rst": ("reStructuredText", Language.RST),
    ".html": ("HTML", Language.HTML),
    ".htm": ("HTML", Language.HTML),
    ".tex": ("LaTeX", Language.LATEX),
    # Language.PERL exists, but from_language raises "not implemented" for it.
    ".pl": ("Perl", None),
    ".pm": ("Perl", None),
    ".sh": ("Shell", None),
    ".bash": ("Shell", None),
    ".css": ("CSS", None),
    ".scss": ("CSS", None),
    ".sql": ("SQL", None),
    ".vue": ("Vue", None),
    ".svelte": ("Svelte", None),
    ".dart": ("Dart", None),
    ".yaml": ("YAML", None),
    ".yml": ("YAML", None),
    ".toml": ("TOML", None),
    ".json": ("JSON", None),
    ".ini": ("INI", None),
    ".cfg": ("INI", None),
    ".txt": ("Text", None),
}
LANGUAGE_BY_FILENAME: dict[str, tuple[str, Language | None]] = {
    "Dockerfile": ("Docker", None),
    "Makefile": ("Make", None),
}


class CloneError(RuntimeError):
    """A repo couldn't be cloned. The message is safe to show to users."""


class IndexingError(RuntimeError):
    """A repo couldn't be indexed. The message is safe to show to users."""


@dataclass(frozen=True)
class SourceFile:
    """A file in a cloned repo that should be indexed."""

    path: Path
    relative_path: str
    language: str
    splitter_language: Language | None


def parse_github_url(url: str) -> str:
    """Return lowercase `owner/repo` for a https://github.com/owner/repo URL, or raise ValueError.

    GitHub names aren't case-sensitive, so lowercasing stops one repo being added twice.
    """
    match = GITHUB_URL_PATTERN.fullmatch(url)
    if match is None or match.group(2) in INVALID_REPO_NAMES:
        raise ValueError(f"Expected a URL like https://github.com/owner/repo, got: {url!r}")
    return f"{match.group(1)}/{match.group(2)}".lower()


def remove_clone(repo_id: int) -> None:
    """Delete a repo's clone folder if it exists, including read-only files."""
    clone = get_settings().repos_dir / str(repo_id)
    if not clone.exists():
        return
    # Git marks some .git files read-only, and Windows refuses to delete read-only files.
    # Links are skipped because chmod would change the file they point to, possibly outside the clone.
    for directory, _, file_names in os.walk(clone):
        os.chmod(directory, stat.S_IRWXU)
        for file_name in file_names:
            path = Path(directory) / file_name
            if not path.is_symlink():
                os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    shutil.rmtree(clone)


def _run_git(command: list[str], url: str) -> None:
    """Run a git command that talks to a repo's server, turning failures into a CloneError."""
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=CLONE_TIMEOUT_SECONDS,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired as error:
        raise CloneError(f"Cloning {url} took longer than {CLONE_TIMEOUT_SECONDS} seconds.") from error
    except subprocess.CalledProcessError as error:
        lines = error.stderr.strip().splitlines()
        detail = lines[-1] if lines else "git gave no details"
        raise CloneError(f"Could not clone {url}. Is it a public repo? ({detail})") from error
    except FileNotFoundError as error:
        raise CloneError("Git is not installed or not on the PATH.") from error


def _fetch_commit(url: str, destination: Path, commit: str) -> None:
    """Download one commit into a new folder, without its history."""
    _run_git(["git", "init", "--quiet", str(destination)], url)
    _run_git(["git", "-C", str(destination), *GIT_NO_LOGIN, "fetch", "--quiet", "--depth", "1", "--", url, commit], url)
    _run_git(["git", "-C", str(destination), "checkout", "--quiet", "FETCH_HEAD"], url)


def clone_repo(url: str, repo_id: int, commit: str | None = None) -> Path:
    """Shallow-clone a repo into data/repos/<repo_id>, replacing any old clone, and return that path.

    With a commit, that exact commit is restored. If GitHub no longer has it, the latest code is
    better than nothing, although citations may then point at lines that have moved.
    """
    remove_clone(repo_id)
    destination = get_settings().repos_dir / str(repo_id)
    if commit:
        try:
            _fetch_commit(url, destination, commit)
            return destination
        except CloneError as error:
            logger.warning("Couldn't restore commit %s, so cloning the latest code instead: %s", commit, error)
            remove_clone(repo_id)
    _run_git(["git", *GIT_NO_LOGIN, "clone", "--depth", "1", "--", url, str(destination)], url)
    return destination


def _current_commit(root: Path) -> str | None:
    """Return the commit a clone is on, or None if the folder isn't a clone."""
    if not (root / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() if result.returncode == 0 else None


def ensure_clone(repo: Repo) -> Path:
    """Return a repo's clone, restoring it first if it's gone.

    Free hosts wipe their disks on every restart, and repos indexed from another machine never had
    a clone here. Vectors survive in Postgres, but the file viewer and guides need the files. The
    lock stops two requests restoring at once, and stops a reader seeing a half-finished clone.
    """
    root = get_settings().repos_dir / str(repo.id)
    with _restore_lock:
        if not root.is_dir():
            logger.info("Restoring the clone of %s.", repo.name)
            clone_repo(repo.url, repo.id, commit=repo.commit_sha)
    return root


def _language_for(path: Path) -> tuple[str, Language | None] | None:
    """Return a file's language details, or None if it isn't a type we index."""
    return LANGUAGE_BY_FILENAME.get(path.name) or LANGUAGE_BY_SUFFIX.get(path.suffix.lower())


def _should_skip_file(path: Path, max_file_bytes: int) -> bool:
    """Return True for symlinks, lockfiles, minified files and oversized files."""
    # Check symlinks first: a repo can link to files outside itself, and later checks would follow the link.
    return (
        path.is_symlink()
        or path.name in LOCKFILES
        or path.name.lower().endswith(MINIFIED_SUFFIXES)
        or path.stat().st_size > max_file_bytes
    )


def iter_source_files(root: Path) -> Iterator[SourceFile]:
    """Yield the files worth indexing under root, in a stable order."""
    max_file_bytes = get_settings().max_file_bytes
    # os.walk doesn't enter symlinked directories by default.
    for directory, dir_names, file_names in os.walk(root):
        # Editing dir_names in place tells os.walk not to enter the removed directories.
        dir_names[:] = sorted(name for name in dir_names if name not in SKIPPED_DIRS)
        for file_name in sorted(file_names):
            path = Path(directory) / file_name
            details = _language_for(path)
            if details is None or _should_skip_file(path, max_file_bytes):
                continue
            yield SourceFile(
                path=path,
                relative_path=path.relative_to(root).as_posix(),
                language=details[0],
                splitter_language=details[1],
            )


def _make_splitter(language: Language | None) -> RecursiveCharacterTextSplitter:
    """Build a splitter for a language.

    from_language loads separators for that language's syntax (`class `, `def `, ...), so chunks
    tend to break between functions, not in the middle of one. add_start_index records each
    chunk's character offset in its metadata, which split_file turns into line numbers.
    """
    settings = get_settings()
    options = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "add_start_index": True,
    }
    if language is None:
        return RecursiveCharacterTextSplitter(**options)
    return RecursiveCharacterTextSplitter.from_language(language, **options)


def split_file(source: SourceFile) -> list[Document]:
    """Split a file into chunks with path, language and 1-based line number metadata.

    A LangChain Document is text (page_content) plus a metadata dict. The vector store keeps both,
    so a search result can say which lines of which file it came from. Files that aren't
    UTF-8 text are skipped.
    """
    try:
        # read_text turns \r\n into \n, so offsets and line numbers agree on every platform.
        text = source.path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    chunks = []
    for chunk in _make_splitter(source.splitter_language).create_documents([text]):
        start_line = text.count("\n", 0, chunk.metadata["start_index"]) + 1
        chunks.append(
            Document(
                # The path helps retrieval match questions like "what does config.py do?".
                # It's added after counting lines, so it can't shift the line numbers.
                # Postgres text can't hold a NUL character, which some files contain.
                page_content=f"# File: {source.relative_path}\n{chunk.page_content}".replace("\x00", ""),
                metadata={
                    "path": source.relative_path,
                    "language": source.language,
                    "start_line": start_line,
                    "end_line": start_line + chunk.page_content.count("\n"),
                },
            )
        )
    return chunks


def _tree_lines(directory: Path, depth: int) -> Iterator[str]:
    """Yield indented tree lines for a directory, folders first."""
    entries = sorted(directory.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower()))
    for entry in entries:
        if entry.is_symlink() or entry.name in SKIPPED_DIRS:
            continue
        indent = "  " * depth
        if entry.is_dir():
            yield f"{indent}{entry.name}/"
            if depth + 1 < MAX_TREE_DEPTH:
                yield from _tree_lines(entry, depth + 1)
        else:
            yield f"{indent}{entry.name}"


def build_file_tree(root: Path) -> str:
    """Return an indented listing of a repo, at most 3 levels deep and 300 lines long."""
    lines = []
    for line in _tree_lines(root, depth=0):
        if len(lines) == MAX_TREE_LINES - 1:
            lines.append("... (truncated)")
            break
        lines.append(line)
    return "\n".join(lines)


def _add_batch_with_waits(store: PGVector, batch: list[Document], repo_name: str) -> None:
    """Embed and store one batch, waiting out rate limits and temporary errors.

    add_documents calls the embedding model first and only stores vectors if that succeeds,
    so repeating a failed batch can't create duplicates.
    """
    settings = get_settings()
    waited = 0.0
    while True:
        try:
            store.add_documents(batch)
            return
        except GEMINI_ERRORS as error:
            if is_daily_quota_error(error):
                raise IndexingError(
                    "Gemini's daily embedding quota is used up. Re-index after it resets, or use a paid key."
                ) from error
            delay = gemini_retry_delay(error, settings.embed_rate_limit_wait_seconds)
            if delay is None:
                raise
            if waited + delay > settings.embed_max_wait_seconds:
                raise IndexingError(
                    f"Gemini kept refusing requests for {waited:.0f} seconds. {describe_gemini_error(error)}"
                ) from error
            logger.warning("%s: waiting %.0fs for Gemini. %s", repo_name, delay, describe_gemini_error(error))
            time.sleep(delay)
            waited += delay


def _embed_chunks(db: Session, repo: Repo, chunks: list[Document]) -> None:
    """Embed all chunks in batches, saving progress on the repo after each one.

    Only Gemini has rate limits, so only Gemini batches get pauses and rate-limit waits.
    """
    settings = get_settings()
    uses_gemini = settings.embedding_provider == "gemini"
    batch_size = settings.embed_batch_size if uses_gemini else LOCAL_EMBED_BATCH_SIZE
    store = get_vectorstore(repo.id)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        if uses_gemini:
            if start > 0:
                time.sleep(settings.embed_delay_seconds)
            _add_batch_with_waits(store, batch, repo.name)
        else:
            store.add_documents(batch)
        repo.chunks_embedded = start + len(batch)
        db.commit()


def collect_chunks(root: Path) -> tuple[list[Document], Counter[str]]:
    """Split a clone's files into chunks, and count the files per language."""
    chunks: list[Document] = []
    languages: Counter[str] = Counter()
    for source in islice(iter_source_files(root), get_settings().max_files):
        file_chunks = split_file(source)
        if file_chunks:
            chunks.extend(file_chunks)
            languages[source.language] += 1
    return chunks, languages


def _run_indexing(db: Session, repo: Repo) -> None:
    """Clone, split and embed a repo, moving its status through cloning, indexing and ready."""
    repo.status = "cloning"
    repo.error = None
    repo.file_count = repo.chunk_count = repo.chunks_embedded = 0
    repo.languages = {}
    repo.embedding_provider = repo.embedding_model = None
    repo.onboarding_guide = None  # it described the old clone
    db.commit()

    root = clone_repo(repo.url, repo.id)
    repo.commit_sha = _current_commit(root)
    repo.status = "indexing"
    db.commit()

    settings = get_settings()
    chunks, languages = collect_chunks(root)
    if not chunks:
        raise IndexingError("This repo has no source files that can be indexed.")
    if settings.demo_mode and len(chunks) > settings.max_index_chunks:
        raise IndexingError(
            f"This repo splits into {len(chunks)} chunks, and this demo's limit is {settings.max_index_chunks}, "
            "to stay within Gemini's daily embedding quota."
        )

    repo.file_count = sum(languages.values())
    repo.chunk_count = len(chunks)
    repo.languages = dict(languages)
    repo.embedding_provider, repo.embedding_model = embedding_identity()
    db.commit()

    delete_collection(repo.id)
    _embed_chunks(db, repo, chunks)
    repo.status = "ready"
    db.commit()


def index_repo(repo_id: int) -> None:
    """Background job: index a repo, and save a readable error if that fails.

    It runs after the request has finished, so it opens its own database session.
    """
    with SessionLocal() as db:
        repo = db.get(Repo, repo_id)
        if repo is None:
            return
        try:
            _run_indexing(db, repo)
            return
        except (CloneError, IndexingError) as error:
            message = str(error)
        except GEMINI_ERRORS as error:
            message = describe_gemini_error(error)
        except ValueError as error:
            # The model factory raises this when a setting is missing from .env.
            message = str(error)
        except Exception:
            logger.exception("Indexing repo %s failed", repo_id)
            message = "Indexing failed because of a server error. See the server log for details."
        db.rollback()
        repo.status = "failed"
        repo.error = message
        db.commit()


def fail_interrupted_repos() -> None:
    """Mark jobs left over from the previous run as failed, so they can be re-indexed.

    Background jobs live in the server process, so none survive a restart. This assumes
    a single server process: with several workers, one could mark another's running job.
    """
    with SessionLocal() as db:
        db.execute(
            update(Repo)
            .where(Repo.status.in_(IN_PROGRESS_STATUSES))
            .values(status="failed", error="Indexing stopped because the server restarted. Re-index to try again.")
        )
        db.commit()
