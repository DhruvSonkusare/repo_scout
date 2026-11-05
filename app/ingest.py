"""Turn a GitHub repo into chunks ready for indexing: clone, filter files, split them."""

import os
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.config import get_settings

CLONE_TIMEOUT_SECONDS = 300
MAX_TREE_DEPTH = 3
MAX_TREE_LINES = 300

GITHUB_URL_PATTERN = re.compile(r"https://github\.com/([A-Za-z0-9_-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?")
INVALID_REPO_NAMES = {".", "..", ".git"}

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


@dataclass(frozen=True)
class SourceFile:
    """A file in a cloned repo that should be indexed."""

    path: Path
    relative_path: str
    language: str
    splitter_language: Language | None


def parse_github_url(url: str) -> str:
    """Return `owner/repo` for a https://github.com/owner/repo URL, or raise ValueError."""
    match = GITHUB_URL_PATTERN.fullmatch(url)
    if match is None or match.group(2) in INVALID_REPO_NAMES:
        raise ValueError(f"Expected a URL like https://github.com/owner/repo, got: {url!r}")
    return f"{match.group(1)}/{match.group(2)}"


def clone_repo(url: str, repo_id: int) -> Path:
    """Shallow-clone a repo into data/repos/<repo_id> and return that path."""
    destination = get_settings().repos_dir / str(repo_id)
    # Cloning runs unattended, so a private or missing repo must fail, not wait for a login:
    # the env var stops terminal prompts, and the empty helper stops Windows login pop-ups.
    command = ["git", "-c", "credential.helper=", "clone", "--depth", "1", "--", url, str(destination)]
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
    return destination


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
                page_content=f"# File: {source.relative_path}\n{chunk.page_content}",
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
