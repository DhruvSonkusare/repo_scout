"""Find and load the source files a guide relies on, so its claims can be checked against real code."""

import ast
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
    found = (_module_file(module, files) for module in modules)
    return list(dict.fromkeys(file for file in found if file is not None and file != path))


def _words(path: str) -> set[str]:
    """Return the meaningful words in a file path."""
    return set(WORD_PATTERN.findall(path.lower())) - COMMON_PATH_WORDS


def _follow_imports(seeds: list[str], files: dict[str, SourceFile]) -> list[str]:
    """Walk from the seed files through their imports, nearest first, up to the file limit.

    Package __init__ files often import every sibling, so files whose paths share words with the
    seeds go first: for scripts/table_parsing.py, that puts the table parsing task ahead of the others.
    """
    seed_words = set().union(*(_words(path) for path in seeds)) if seeds else set()
    chosen: list[str] = []
    seen = set(seeds)
    level = seeds
    for _ in range(MAX_IMPORT_DEPTH + 1):
        next_level: list[str] = []
        for path in level:
            if len(chosen) == MAX_VERIFY_FILES:
                return chosen
            chosen.append(path)
            for imported in _imported_files(path, _read(files[path]), files):
                if imported not in seen:
                    seen.add(imported)
                    next_level.append(imported)
        level = sorted(next_level, key=lambda path: len(_words(path) & seed_words), reverse=True)
    return chosen


def load_guide_sources(root: Path, guide: str, excerpt_paths: list[str]) -> list[LoadedFile]:
    """Load the files a guide's claims rest on: what it cites, then what those import.

    A guide that cites nothing is checked against the code files its excerpts came from.
    """
    files = _source_files(root)
    seeds = list(dict.fromkeys(_cited_files(guide, files) + _configs_mentioning_paths(guide, files)))
    if not seeds:
        seeds = list(dict.fromkeys(path for path in excerpt_paths if path in files))
    loaded: list[LoadedFile] = []
    for path in _follow_imports(seeds, files):
        text = _read(files[path])
        cut = text[:MAX_VERIFY_CHARS_PER_FILE] + TRUNCATED_NOTE if len(text) > MAX_VERIFY_CHARS_PER_FILE else text
        loaded.append(LoadedFile(path, cut))
    return loaded


def format_source_files(files: list[LoadedFile]) -> str:
    """Describe the loaded files for a prompt. Four backticks, because a file can contain three."""
    return "\n\n".join(f"### {file.path}\n````\n{file.text}\n````" for file in files)
