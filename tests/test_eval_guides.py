"""Tests for the guide eval: the guide that misled us must fail, and a correct one must pass."""

from pathlib import Path

import pytest

from scripts.eval_guides import CASES, evaluate, main, verdict_of

CASE = CASES[0]
FACT_NAMES = [fact.name for fact in CASE.facts]

# The parts of the real guide that were wrong, word for word. The code rejects PDFs in this step.
MISLEADING_GUIDE = """## Does it solve your problem?
**Fully**. The toolkit includes a dedicated table recognition module (`StructEqTable` and `PaddleOCR+TableMaster`) specifically designed to convert table images from PDFs into structured formats like LaTeX, HTML, or Markdown.

## Step-by-step: solving YOUR problem with this repo
1. **Configure the Table Parsing Task**:
   Modify `configs/table_parsing.yaml` to point to your input PDF or image directory. The toolkit supports single files or entire directories.

## Gotchas and next steps
* **OCR Dependencies**: Table recognition relies on OCR. Ensure your `configs/table_parsing.yaml` correctly points to the PaddleOCR model paths (e.g., `models/OCR/PaddleOCR/det/ch_PP-OCRv4_det`).
"""

CORRECT_GUIDE = """## Does it solve your problem?
**Partly**. The table parsing task only accepts .png, .jpg and .jpeg images, so PDFs are rejected.

## Step-by-step: solving YOUR problem with this repo
1. Convert each PDF page to an image first, then point `configs/table_parsing.yaml` at that folder.

## Gotchas and next steps
* PaddleOCR model paths such as `models/OCR/PaddleOCR/det/ch_PP-OCRv4_det` are set in `configs/ocr.yaml`.
"""


def results(guide: str) -> dict[str, bool]:
    return dict(evaluate(guide, CASE))


def test_the_case_covers_the_facts_we_checked_by_hand() -> None:
    assert CASE.repo == "opendatalab/pdf-extract-kit"
    assert CASE.problem == "extract tables from scanned PDFs in Python"
    assert len(FACT_NAMES) == 4


def test_the_misleading_guide_fails_every_fact() -> None:
    assert list(results(MISLEADING_GUIDE).values()) == [False, False, False, False]


def test_a_correct_guide_passes_every_fact() -> None:
    assert list(results(CORRECT_GUIDE).values()) == [True, True, True, True]


def test_a_clear_statement_about_images_passes_without_partly() -> None:
    guide = "## Does it solve your problem?\n**No**. The input must be images, as PDFs are rejected."

    assert results(guide)[FACT_NAMES[0]] is True


def test_only_some_formats_fails_the_format_fact() -> None:
    assert results("It reads PNG and JPG files.")[FACT_NAMES[1]] is False


def test_saying_not_to_pass_a_pdf_is_fine() -> None:
    assert results("Don't point it at a PDF, as it would be rejected.")[FACT_NAMES[2]] is True


def test_saying_paddleocr_is_not_in_that_config_is_fine() -> None:
    guide = "PaddleOCR isn't set in `configs/table_parsing.yaml`, but in `configs/ocr.yaml`."

    assert results(guide)[FACT_NAMES[3]] is True


def test_paddleocr_paths_without_their_config_fail() -> None:
    assert results("Download to `models/OCR/PaddleOCR/det`.")[FACT_NAMES[3]] is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("## Does it solve your problem?\n**Partly**. Reasons.", "Partly"),
        ("## Does it solve your problem?\n\nFully, because.", "Fully"),
        ("No verdict here.", None),
    ],
)
def test_the_verdict_is_read(text: str, expected: str | None) -> None:
    assert verdict_of(text) == expected


@pytest.mark.parametrize(("guide", "code"), [(MISLEADING_GUIDE, 1), (CORRECT_GUIDE, 0)])
def test_the_script_reports_each_fact_and_exits_by_result(
    guide: str, code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    file = tmp_path / "guide.md"
    file.write_text(guide, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["eval_guides.py", "--guide-file", str(file)])

    assert main() == code

    output = capsys.readouterr().out
    assert all(name in output for name in FACT_NAMES)
    assert ("FAIL" in output) is (code == 1)


def test_generating_needs_an_indexed_repo(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["eval_guides.py", "--generate"])

    assert main() == 2
    assert "isn't indexed" in capsys.readouterr().out
