"""Find and load the source files a guide relies on, so its claims can be checked against real code."""

import ast
import heapq
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

from app.ingest import SourceFile, iter_source_files

# Enough for a script, what it imports and the base classes those use, without flooding the prompt.
MAX_VERIFY_FILES = 15
MAX_VERIFY_CHARS_PER_FILE = 6000
MAX_IMPORT_DEPTH = 4
MAX_FILES_MENTIONING_A_PATH = 3
TRUNCATED_NOTE = "\n... (the rest of this file was cut)"

# Documents can describe code that's out of date, so only other files count as evidence.
DOC_LANGUAGES = {"Markdown", "reStructuredText", "HTML", "LaTeX", "Text"}
CONFIG_LANGUAGES = {"YAML", "TOML", "JSON", "INI"}
INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
PATH_PATTERN = re.compile(r"[\w.-]+(?:/[\w.-]+)+")
WORD_PATTERN = re.compile(r"[a-z0-9]+")
COMMON_PATH_WORDS = {"py", "yaml", "yml", "json", "scripts", "script", "configs", "config", "src", "main"}


@dataclass(frozen=True)
class LoadedFile:
    """A source file's path and text, cut to the size limit."""

    path: str
    text: str


def is_built_doc(path: str) -> bool:
    """Return True for generated documentation, which is noisy and can be older than the code."""
    parts = path.lower().split("/")
    return "_build" in parts or (parts[0] in {"doc", "docs"} and parts[-1].endswith((".html", ".htm")))


def _source_files(root: Path) -> dict[str, SourceFile]:
    """Return the repo's code and config files by relative path, leaving out documents."""
    return {
        source.relative_path: source
        for source in iter_source_files(root)
        if source.language not in DOC_LANGUAGES and not is_built_doc(source.relative_path)
    }


def _cited_files(guide: str, files: dict[str, SourceFile]) -> list[str]:
    """Return the files the guide names, in the order it first names them."""
    positions: dict[str, int] = {}
    for path in files:
        match = re.search(rf"(?<![\w/.-]){re.escape(path)}(?![\w/-])", guide)
        if match is not None:
            positions[path] = match.start()
    return sorted(positions, key=lambda path: positions[path])


def _read(source: SourceFile) -> str:
    """Return a file's text. Repos can hold files in any encoding, and a few odd characters don't matter here."""
    return source.path.read_text(encoding="utf-8", errors="replace")


def _configs_mentioning_paths(guide: str, files: dict[str, SourceFile]) -> list[str]:
    """Return the config files that define paths the guide names but the repo doesn't contain.

    A path like models/det is created at runtime, so no file has that name. The way to check a
    claim about it is to find the config that really sets it.
    """
    missing = {
        path.rstrip("/")
        for code in INLINE_CODE_PATTERN.findall(guide)
        for path in PATH_PATTERN.findall(code)
        if path.rstrip("/") not in files
    }
    if not missing:
        return []
    configs = [path for path, source in files.items() if source.language in CONFIG_LANGUAGES]
    texts = {path: _read(files[path]) for path in configs}
    found: list[str] = []
    for path in sorted(missing):
        found += [config for config, text in texts.items() if path in text][:MAX_FILES_MENTIONING_A_PATH]
    return list(dict.fromkeys(found))


def _module_file(module: str, files: dict[str, SourceFile]) -> str | None:
    """Return the file holding a dotted module name, if it's in this repo."""
    path = module.replace(".", "/")
    for candidate in (f"{path}.py", f"{path}/__init__.py", f"src/{path}.py", f"src/{path}/__init__.py"):
        if candidate in files:
            return candidate
    return None


def _imported_files(path: str, text: str, files: dict[str, SourceFile]) -> list[str]:
    """Return the repo's own Python files that a Python file imports. Other languages have none."""
    if not path.endswith(".py"):
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    package = path.split("/")[:-1]
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # level counts the leading dots: 1 is this file's package, 2 is its parent, and so on.
            parent = package[: len(package) - node.level + 1] if node.level else []
            base = ".".join([*parent, *(node.module.split(".") if node.module else [])])
            modules += [base] if base else []
            # `from package import module` names a module, while `from module import name` doesn't.
            modules += [f"{base}.{alias.name}" if base else alias.name for alias in node.names]
    # Importing a.b.c also runs a/__init__.py and a/b/__init__.py, and those often import more.
    parents = (".".join(module.split(".")[:end]) for module in modules for end in range(1, module.count(".") + 2))
    found = (_module_file(module, files) for module in parents)
    return list(dict.fromkeys(file for file in found if file is not None and file != path))


def _words(path: str) -> set[str]:
    """Return the meaningful words in a file path."""
    return set(WORD_PATTERN.findall(path.lower())) - COMMON_PATH_WORDS


def _scripts_for_configs(cited: list[str], files: dict[str, SourceFile]) -> list[str]:
    """Return the Python files named like a cited config, as configs/tables.yaml usually belongs to scripts/tables.py."""
    stems = {Path(path).stem for path in cited if files[path].language in CONFIG_LANGUAGES}
    return [path for path in files if path.endswith(".py") and Path(path).stem in stems]


def _follow_imports(seeds: list[str], related_to: list[str], files: dict[str, SourceFile]) -> list[str]:
    """Walk from the seed files through their imports, up to the file limit, closest related first.

    Package __init__ files often import every sibling, so a walk by distance alone would fill the
    limit with unrelated files. So a file ranks by the words its path shares with related_to, and a
    file imported by a related file ranks with it. For configs/table_parsing.yaml, that reaches the
    table parsing model and its base class before the other tasks. Empty files don't use up the limit.
    """
    words_to_match = set().union(*(_words(path) for path in related_to)) if related_to else set()

    def shared_words(path: str) -> int:
        return len(_words(path) & words_to_match)

    order = itertools.count()
    # Each entry sorts by: seeds first, inherited rank, the file's own rank, import depth, order found.
    queue = [(0, 0, 0, 0, next(order), path) for path in seeds]
    heapq.heapify(queue)
    best_rank = {path: 0 for path in seeds}
    visited: set[str] = set()
    chosen: list[str] = []
    while queue and len(chosen) < MAX_VERIFY_FILES:
        _, _, _, depth, _, path = heapq.heappop(queue)
        if path in visited:
            continue
        visited.add(path)
        text = _read(files[path])
        if text.strip():
            chosen.append(path)
        if depth == MAX_IMPORT_DEPTH:
            continue
        # A seed passes on no rank, because a cited script imports plenty that has nothing to do with it.
        own_rank = 0 if path in seeds else shared_words(path)
        for imported in _imported_files(path, text, files):
            rank = max(shared_words(imported), own_rank)
            if imported not in visited and rank > best_rank.get(imported, -1):
                # A better route to a queued file adds a new entry, and the old one is skipped later.
                best_rank[imported] = rank
                heapq.heappush(queue, (1, -rank, -shared_words(imported), depth + 1, next(order), imported))
    return chosen


def load_guide_sources(root: Path, guide: str, excerpt_paths: list[str]) -> list[LoadedFile]:
    """Load the files a guide's claims rest on: what it cites, the scripts its configs belong to, then what those import.

    A guide that cites nothing is checked against the code files its excerpts came from.
    """
    files = _source_files(root)
    cited = _cited_files(guide, files)
    seeds = list(dict.fromkeys(cited + _scripts_for_configs(cited, files) + _configs_mentioning_paths(guide, files)))
    if not seeds:
        seeds = cited = list(dict.fromkeys(path for path in excerpt_paths if path in files))
    loaded: list[LoadedFile] = []
    for path in _follow_imports(seeds, cited, files):
        text = _read(files[path])
        cut = text[:MAX_VERIFY_CHARS_PER_FILE] + TRUNCATED_NOTE if len(text) > MAX_VERIFY_CHARS_PER_FILE else text
        loaded.append(LoadedFile(path, cut))
    return loaded


def format_source_files(files: list[LoadedFile]) -> str:
    """Describe the loaded files for a prompt. Four backticks, because a file can contain three."""
    return "\n\n".join(f"### {file.path}\n````\n{file.text}\n````" for file in files)
