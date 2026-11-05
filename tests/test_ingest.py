"""Tests for repo ingestion, using temporary directories and a local git repo."""

import subprocess
from pathlib import Path

import pytest
from langchain_text_splitters import Language

from app.ingest import (
    LANGUAGE_BY_SUFFIX,
    CloneError,
    SourceFile,
    build_file_tree,
    clone_repo,
    iter_source_files,
    parse_github_url,
    split_file,
)

pytestmark = pytest.mark.usefixtures("fresh_settings")


def write_file(root: Path, relative_path: str, content: str = "x = 1\n") -> Path:
    """Create a file, along with any parent folders."""
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path


def indexed_paths(root: Path) -> list[str]:
    """Return the relative paths iter_source_files finds."""
    return [source.relative_path for source in iter_source_files(root)]


def python_source(path: Path) -> SourceFile:
    """Describe a Python file for split_file."""
    return SourceFile(path=path, relative_path=path.name, language="Python", splitter_language=Language.PYTHON)


def long_python_file(function_count: int = 60) -> str:
    """Build Python source long enough to need several chunks, with blank lines and indentation."""
    functions = [
        f"def function_{number}(value):\n"
        f'    """Docstring for function {number}."""\n'
        "\n"
        f"    result = value * {number}\n"
        f"    return result + {number}\n"
        for number in range(function_count)
    ]
    return "\n\n".join(functions)


def body_of(page_content: str) -> str:
    """Remove the `# File:` line from a chunk."""
    return page_content.split("\n", 1)[1]


# --- parse_github_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/pallets/click", "pallets/click"),
        ("https://github.com/pallets/click/", "pallets/click"),
        ("https://github.com/pallets/click.git", "pallets/click"),
        ("https://github.com/some-org/repo_name.js", "some-org/repo_name.js"),
        ("https://github.com/acme/.github", "acme/.github"),
    ],
)
def test_parse_github_url_accepts_repo_urls(url: str, expected: str) -> None:
    assert parse_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "pallets/click",
        "github.com/pallets/click",
        "http://github.com/pallets/click",
        "git@github.com:pallets/click.git",
        "https://gitlab.com/pallets/click",
        "https://github.com.evil.example/pallets/click",
        "https://github.com/pallets",
        "https://github.com/pallets/click/tree/main",
        "https://github.com/pallets/click?tab=readme",
        "https://github.com/pallets/click#readme",
        "https://github.com/pallets/..",
        "https://github.com/pallets/.git",
        "https://github.com/pallets/click\n",
        "https://github.com/pallets/click --upload-pack=evil",
    ],
)
def test_parse_github_url_rejects_everything_else(url: str) -> None:
    with pytest.raises(ValueError, match="https://github.com/owner/repo"):
        parse_github_url(url)


# --- clone_repo


def test_clone_repo_makes_a_shallow_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    origin = tmp_path / "origin"
    write_file(origin, "hello.py")
    git = ["git", "-C", str(origin), "-c", "user.name=Test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", "init", "--quiet", str(origin)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "--quiet", "-m", "first"], check=True)
    subprocess.run([*git, "commit", "--quiet", "--allow-empty", "-m", "second"], check=True)

    clone = clone_repo(origin.as_uri(), repo_id=7)

    assert clone == tmp_path / "data" / "repos" / "7"
    assert (clone / "hello.py").is_file()
    history = subprocess.run(
        ["git", "-C", str(clone), "rev-list", "--count", "HEAD"], check=True, capture_output=True, text=True
    )
    assert history.stdout.strip() == "1"


def test_clone_repo_explains_a_failed_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(CloneError, match="Could not clone"):
        clone_repo((tmp_path / "missing").as_uri(), repo_id=1)


def test_clone_repo_explains_a_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    def time_out(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(subprocess, "run", time_out)

    with pytest.raises(CloneError, match="took longer"):
        clone_repo("https://github.com/pallets/click", repo_id=1)


# --- iter_source_files


def test_iter_source_files_finds_source_with_language_details(tmp_path: Path) -> None:
    write_file(tmp_path, "src/app.py")
    write_file(tmp_path, "web/App.TSX")
    write_file(tmp_path, "Dockerfile")

    sources = {source.relative_path: source for source in iter_source_files(tmp_path)}

    assert set(sources) == {"Dockerfile", "src/app.py", "web/App.TSX"}
    assert sources["src/app.py"].language == "Python"
    assert sources["src/app.py"].splitter_language == Language.PYTHON
    assert sources["web/App.TSX"].splitter_language == Language.TS
    assert sources["Dockerfile"].splitter_language is None


@pytest.mark.parametrize(
    "skipped",
    [
        "node_modules/lib/index.js",
        ".git/hooks/pre-commit.py",
        "venv/lib/site.py",
        ".venv/lib/site.py",
        "dist/app.js",
        "build/app.js",
        "src/__pycache__/app.py",
        "package-lock.json",
        "yarn.lock",
        "poetry.lock",
        "web/go.sum",
        "static/app.min.js",
        "static/style.min.css",
        "logo.png",
        "archive.zip",
    ],
)
def test_iter_source_files_skips_unwanted_paths(tmp_path: Path, skipped: str) -> None:
    write_file(tmp_path, "keep.py")
    write_file(tmp_path, skipped)

    assert indexed_paths(tmp_path) == ["keep.py"]


def test_iter_source_files_applies_the_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_BYTES", "10")
    write_file(tmp_path, "exactly_at_limit.py", "x" * 10)
    write_file(tmp_path, "over_limit.py", "x" * 11)

    assert indexed_paths(tmp_path) == ["exactly_at_limit.py"]


def test_iter_source_files_skips_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    write_file(outside, "secret.py")
    repo = tmp_path / "repo"
    write_file(repo, "keep.py")
    try:
        (repo / "link.py").symlink_to(outside / "secret.py")
        (repo / "broken.py").symlink_to(outside / "does_not_exist.py")
        (repo / "linked_dir").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This machine doesn't allow creating symlinks.")

    assert indexed_paths(repo) == ["keep.py"]


def test_symlinks_are_skipped_even_where_they_cannot_be_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Creating real symlinks needs special rights on Windows, so this pretends one file is a link.
    write_file(tmp_path, "keep.py")
    write_file(tmp_path, "link.py")
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.name == "link.py")

    assert indexed_paths(tmp_path) == ["keep.py"]
    assert build_file_tree(tmp_path) == "keep.py"


@pytest.mark.parametrize("suffix", sorted(LANGUAGE_BY_SUFFIX))
def test_every_mapped_suffix_can_be_split(tmp_path: Path, suffix: str) -> None:
    write_file(tmp_path, f"sample{suffix}", "first line\nsecond line\n")

    source = next(iter_source_files(tmp_path))

    assert split_file(source)


# --- split_file


def test_split_file_line_numbers_match_the_real_lines(tmp_path: Path) -> None:
    text = long_python_file()
    path = write_file(tmp_path, "sample.py", text)
    lines = text.split("\n")

    chunks = split_file(python_source(path))

    assert len(chunks) > 1
    for chunk in chunks:
        cited = lines[chunk.metadata["start_line"] - 1 : chunk.metadata["end_line"]]
        assert "\n".join(cited).strip() == body_of(chunk.page_content)


def test_split_file_finds_a_known_function_on_its_known_line(tmp_path: Path) -> None:
    text = long_python_file()
    path = write_file(tmp_path, "sample.py", text)
    expected_line = text.split("\n").index("def function_40(value):") + 1

    chunks = split_file(python_source(path))

    chunk = next(chunk for chunk in chunks if "def function_40(value):" in chunk.page_content)
    offset = body_of(chunk.page_content).split("\n").index("def function_40(value):")
    assert chunk.metadata["start_line"] + offset == expected_line


def test_split_file_line_numbers_are_right_with_windows_line_endings(tmp_path: Path) -> None:
    text = long_python_file()
    path = write_file(tmp_path, "sample.py", text.replace("\n", "\r\n"))
    lines = text.split("\n")

    chunks = split_file(python_source(path))

    for chunk in chunks:
        cited = lines[chunk.metadata["start_line"] - 1 : chunk.metadata["end_line"]]
        assert "\n".join(cited).strip() == body_of(chunk.page_content)


def test_split_file_line_numbers_are_right_with_repeated_code(tmp_path: Path) -> None:
    block = "def same():\n    return 1\n"
    text = "\n".join([block] * 300)
    path = write_file(tmp_path, "sample.py", text)
    lines = text.split("\n")

    chunks = split_file(python_source(path))

    assert len(chunks) > 1
    for chunk in chunks:
        cited = lines[chunk.metadata["start_line"] - 1 : chunk.metadata["end_line"]]
        assert "\n".join(cited).strip() == body_of(chunk.page_content)


def test_split_file_starts_chunks_at_function_boundaries(tmp_path: Path) -> None:
    path = write_file(tmp_path, "sample.py", long_python_file())

    chunks = split_file(python_source(path))

    assert all(body_of(chunk.page_content).startswith("def ") for chunk in chunks)


def test_split_file_respects_chunk_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    path = write_file(tmp_path, "sample.py", long_python_file())

    chunks = split_file(python_source(path))

    assert all(len(body_of(chunk.page_content)) <= 200 for chunk in chunks)


def test_split_file_adds_path_prefix_and_metadata(tmp_path: Path) -> None:
    path = write_file(tmp_path, "src/app.py", "print('hello')\n")
    source = SourceFile(path=path, relative_path="src/app.py", language="Python", splitter_language=Language.PYTHON)

    chunks = split_file(source)

    assert chunks[0].page_content == "# File: src/app.py\nprint('hello')"
    assert chunks[0].metadata == {"path": "src/app.py", "language": "Python", "start_line": 1, "end_line": 1}


def test_split_file_uses_the_generic_splitter_for_other_file_types(tmp_path: Path) -> None:
    path = write_file(tmp_path, "config.yaml", "name: demo\n")
    source = SourceFile(path=path, relative_path="config.yaml", language="YAML", splitter_language=None)

    assert body_of(split_file(source)[0].page_content) == "name: demo"


def test_split_file_returns_nothing_for_empty_and_binary_files(tmp_path: Path) -> None:
    empty = write_file(tmp_path, "empty.py", "")
    binary = tmp_path / "binary.py"
    binary.write_bytes(b"\xff\xfe\x00\x01")

    assert split_file(python_source(empty)) == []
    assert split_file(python_source(binary)) == []


# --- build_file_tree


def test_build_file_tree_lists_folders_first_and_skips_unwanted_dirs(tmp_path: Path) -> None:
    write_file(tmp_path, "README.md")
    write_file(tmp_path, "src/app.py")
    write_file(tmp_path, "node_modules/lib/index.js")

    assert build_file_tree(tmp_path) == "src/\n  app.py\nREADME.md"


def test_build_file_tree_stops_at_depth_three(tmp_path: Path) -> None:
    write_file(tmp_path, "one/two/three/four/deep.py")

    assert build_file_tree(tmp_path) == "one/\n  two/\n    three/"


def test_build_file_tree_is_capped_at_300_lines(tmp_path: Path) -> None:
    for number in range(350):
        write_file(tmp_path, f"file_{number:03}.py")

    lines = build_file_tree(tmp_path).split("\n")

    assert len(lines) == 300
    assert lines[-1] == "... (truncated)"


def test_build_file_tree_is_not_marked_truncated_when_it_fits(tmp_path: Path) -> None:
    for number in range(299):
        write_file(tmp_path, f"file_{number:03}.py")

    lines = build_file_tree(tmp_path).split("\n")

    assert len(lines) == 299
    assert "... (truncated)" not in lines
