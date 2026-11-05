"""Tests for finding the source files a guide relies on, using small repos in a temporary folder."""

from pathlib import Path

import pytest

from app.finder.source_context import (
    MAX_VERIFY_CHARS_PER_FILE,
    MAX_VERIFY_FILES,
    TRUNCATED_NOTE,
    LoadedFile,
    format_source_files,
    is_built_doc,
    load_guide_sources,
)

pytestmark = pytest.mark.usefixtures("fresh_settings")


def write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def paths_for(root: Path, guide: str, excerpts: list[str] | None = None) -> list[str]:
    return [file.path for file in load_guide_sources(root, guide, excerpts or [])]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("docs/_build/html/code.html", True),
        ("docs/zh_cn/_build/doctrees/x.txt", True),
        ("docs/index.html", True),
        ("doc/page.HTM", True),
        ("docs/guide.rst", False),
        ("templates/index.html", False),
        ("src/build.py", False),
    ],
)
def test_built_docs_are_recognized(path: str, expected: bool) -> None:
    assert is_built_doc(path) is expected


def test_files_named_in_the_guide_are_loaded_in_order(tmp_path: Path) -> None:
    write(tmp_path, {"a.py": "x = 1", "b/c.py": "y = 2", "other.py": "z = 3"})

    assert paths_for(tmp_path, "Run `b/c.py`, then `a.py`.") == ["b/c.py", "a.py"]


def test_a_longer_path_that_contains_a_file_name_does_not_count(tmp_path: Path) -> None:
    write(tmp_path, {"a.py": "x = 1", "sub/a.py": "y = 2"})

    assert paths_for(tmp_path, "See `sub/a.py`.") == ["sub/a.py"]


def test_documents_and_built_docs_are_not_evidence(tmp_path: Path) -> None:
    write(tmp_path, {"README.md": "docs", "docs/_build/x.py": "built", "a.py": "x = 1"})

    assert paths_for(tmp_path, "See `README.md`, `docs/_build/x.py` and `a.py`.") == ["a.py"]


def test_imports_are_followed_through_packages_and_relative_imports(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "scripts/run.py": "from pkg.tools import loader\nimport pkg.base",
            "pkg/__init__.py": "",
            "pkg/tools/__init__.py": "",
            "pkg/tools/loader.py": "from ..base import Base\nfrom . import helper",
            "pkg/tools/helper.py": "",
            "pkg/base.py": "class Base: ...",
            "pkg/unused.py": "",
        },
    )

    paths = paths_for(tmp_path, "Run `scripts/run.py`.")

    assert paths[0] == "scripts/run.py"
    assert set(paths) == {
        "scripts/run.py",
        "pkg/tools/__init__.py",
        "pkg/tools/loader.py",
        "pkg/base.py",
        "pkg/tools/helper.py",
    }


def test_a_src_layout_is_followed(tmp_path: Path) -> None:
    write(tmp_path, {"run.py": "import pkg.mod", "src/pkg/mod.py": ""})

    assert paths_for(tmp_path, "Run `run.py`.") == ["run.py", "src/pkg/mod.py"]


def test_outside_packages_and_broken_files_are_harmless(tmp_path: Path) -> None:
    write(tmp_path, {"run.py": "import os\nimport requests", "broken.py": "def ("})

    assert paths_for(tmp_path, "See `run.py` and `broken.py`.") == ["run.py", "broken.py"]


def test_files_related_to_the_seed_win_when_the_limit_is_reached(tmp_path: Path) -> None:
    others = {f"pkg/other_{number}.py": "" for number in range(MAX_VERIFY_FILES)}
    imports = "\n".join(f"from pkg import other_{number}" for number in range(MAX_VERIFY_FILES))
    write(tmp_path, {"run_tables.py": f"{imports}\nfrom pkg import tables", "pkg/tables.py": "", **others})

    paths = paths_for(tmp_path, "Run `run_tables.py`.")

    assert len(paths) == MAX_VERIFY_FILES
    assert paths[1] == "pkg/tables.py"


def test_a_config_defining_a_runtime_path_is_loaded(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "configs/ocr.yaml": "model: models/det/v4",
            "configs/tables.yaml": "output: out",
            "run.py": "x = 1",
        },
    )

    paths = paths_for(tmp_path, "See `configs/tables.yaml`, which sets `models/det/v4`.")

    assert paths == ["configs/tables.yaml", "configs/ocr.yaml"]


def test_a_guide_naming_nothing_falls_back_to_the_excerpts(tmp_path: Path) -> None:
    write(tmp_path, {"a.py": "x = 1", "README.md": "docs"})

    assert paths_for(tmp_path, "No files here.", ["README.md", "a.py", "a.py", "gone.py"]) == ["a.py"]


def test_nothing_to_load_returns_an_empty_list(tmp_path: Path) -> None:
    write(tmp_path, {"README.md": "docs"})

    assert paths_for(tmp_path, "No files here.") == []


def test_long_files_are_cut_and_say_so(tmp_path: Path) -> None:
    write(tmp_path, {"long.py": "x" * (MAX_VERIFY_CHARS_PER_FILE + 50), "short.py": "y"})

    loaded = load_guide_sources(tmp_path, "See `long.py` and `short.py`.", [])

    assert loaded[0].text == "x" * MAX_VERIFY_CHARS_PER_FILE + TRUNCATED_NOTE
    assert loaded[1].text == "y"


def test_a_linked_file_is_not_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Creating real symlinks needs special rights on Windows, so this pretends the file is a link.
    write(tmp_path, {"secret.py": "from outside the clone", "a.py": "x = 1"})
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.name == "secret.py")

    assert paths_for(tmp_path, "See `secret.py` and `a.py`.") == ["a.py"]


def test_format_shows_each_path_and_its_text() -> None:
    text = format_source_files([LoadedFile("a.py", "x = 1"), LoadedFile("b.py", "y = 2")])

    assert text == "### a.py\n````\nx = 1\n````\n\n### b.py\n````\ny = 2\n````"
