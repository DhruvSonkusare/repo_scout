"""Tests for the guide eval: the guide that misled us must fail, and a correct one must pass."""

from pathlib import Path

import pytest

from scripts.eval_guides import CASES, evaluate, main, verdict_of

CASE = CASES[0]
FACT_NAMES = [fact.name for fact in CASE.facts]
# A full guide from before verification existed, kept as a permanent bad example.
BEFORE_FIX_GUIDE = Path(__file__).parent / "fixtures" / "guides" / "pdf-extract-kit-before.md"
# The first verified guide: it fixed the main error, but lost the download and doubted a real file.
VERIFIED_V1_GUIDE = BEFORE_FIX_GUIDE.with_name("pdf-extract-kit-verified-v1.md")

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

## Setup
1. Install with `pip install -r requirements-cpu.txt` on a CPU.
2. Download the model weights from Hugging Face.

## Step-by-step: solving YOUR problem with this repo
1. Save each PDF page as a PNG image file first, then point `configs/table_parsing.yaml` at that folder.

## Gotchas and next steps
* PaddleOCR model paths such as `models/OCR/PaddleOCR/det/ch_PP-OCRv4_det` are set in `configs/ocr.yaml`.
"""


def results(guide: str) -> dict[str, bool]:
    return dict(evaluate(guide, CASE))


def test_the_case_covers_the_facts_we_checked_by_hand() -> None:
    assert CASE.repo == "opendatalab/pdf-extract-kit"
    assert CASE.problem == "extract tables from scanned PDFs in Python"
    assert len(FACT_NAMES) == 7


def test_the_misleading_excerpts_fail_every_fact_they_cover() -> None:
    assert list(results(MISLEADING_GUIDE).values()) == [False, False, False, False, False, False, True]


def test_the_real_guide_that_misled_us_only_passes_the_setup_facts() -> None:
    passed = list(results(BEFORE_FIX_GUIDE.read_text(encoding="utf-8")).values())

    assert passed == [False, False, False, False, False, True, True]


def test_the_first_verified_guide_fails_the_gaps_found_by_reading_it() -> None:
    passed = list(results(VERIFIED_V1_GUIDE.read_text(encoding="utf-8")).values())

    assert passed == [True, True, True, True, False, False, False]


def test_a_correct_guide_passes_every_fact() -> None:
    assert list(results(CORRECT_GUIDE).values()) == [True] * 7


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


def test_discussing_paddleocr_without_its_config_fails() -> None:
    assert results("It can use PaddleOCR+TableMaster for tables.")[FACT_NAMES[3]] is False


def test_never_mentioning_paddleocr_is_fine() -> None:
    assert results("It uses another model.")[FACT_NAMES[3]] is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Save each page as a PNG file.", True),
        ("Call `image.save(path)` on each page.", True),
        ("Write the pages to a folder.", True),
        ("Convert each page to an image in memory.", False),
    ],
)
def test_saving_pages_as_files_is_recognized(text: str, expected: bool) -> None:
    assert results(text)[FACT_NAMES[4]] is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Download the model weights from Hugging Face.", True),
        ("The checkpoints have to be downloaded first.", True),
        ("Download the repo as a zip.", False),
        ("Install the requirements.", False),
    ],
)
def test_the_model_download_is_recognized(text: str, expected: bool) -> None:
    assert results(text)[FACT_NAMES[5]] is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Use `requirements-cpu.txt` on a CPU.", True),
        ("The existence of `requirements-cpu.txt` is not confirmed by the source code.", False),
        ("`requirements-cpu.txt` may not exist.", False),
    ],
)
def test_doubt_about_the_cpu_requirements_fails(text: str, expected: bool) -> None:
    assert results(text)[FACT_NAMES[6]] is expected


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
