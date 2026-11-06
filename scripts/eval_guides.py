"""Check problem guides against facts we verified by reading the repo's code, and report pass or fail per fact.

Check a saved guide with no model calls:   python scripts/eval_guides.py --guide-file guide.md
Write a new guide and check it:            python scripts/eval_guides.py --generate
Generating needs the repo indexed, and makes real model calls: one for the draft and one for its verification.
"""

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Running this file directly puts scripts/ on the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NEGATION = re.compile(r"\b(not|never|no|only|must|instead|first|convert\w*|render\w*|reject\w*|unsupported)\b|n't", re.I)
_IMAGE = r"\b(images?|png|jpe?g)\b"
# Merely mentioning images isn't enough: "converts table images from PDFs" says nothing about what's accepted.
IMAGES_ONLY = re.compile(
    rf"\bonly\b.{{0,60}}{_IMAGE}|{_IMAGE}.{{0,40}}\bonly\b|\b(must be|requires?|needs?)\b.{{0,40}}{_IMAGE}"
    r"|\b(rejects?|unsupported|not (accept|support)|n't (accept|support))\b.{0,40}\bpdfs?\b"
    r"|\bpdfs?\b.{0,40}\b(rejected|unsupported|not (accepted|supported))\b",
    re.I,
)
SAVES_IMAGES = re.compile(r"\.save\(|\b(sav(e|es|ing)|writ(e|es|ing)|export(s|ing)?)\b.{0,60}\b(images?|png|jpe?g|pages?)\b", re.I)
_WEIGHTS = r"\b(models?|weights?|checkpoints?)\b"
MODEL_DOWNLOAD = re.compile(rf"\bdownload\w*\b.{{0,80}}{_WEIGHTS}|{_WEIGHTS}.{{0,80}}\bdownload\w*\b|snapshot_download", re.I)
DOUBT = re.compile(r"not confirmed|unconfirmed|(can't|cannot) confirm|(doesn't|does not|may not) exist|not found|missing", re.I)
VERDICT = re.compile(r"## Does it solve your problem\?\s*\**\s*(Fully|Partly|No)\b", re.I)


@dataclass(frozen=True)
class Fact:
    """One thing a good guide must get right. The check reads the guide's text."""

    name: str
    check: Callable[[str], bool]


@dataclass(frozen=True)
class GuideCase:
    """A problem, the repo to solve it with, and the facts its guide must get right."""

    problem: str
    repo: str
    description: str
    facts: list[Fact]


def verdict_of(guide: str) -> str | None:
    """Return the guide's verdict as Fully, Partly or No, if it states one."""
    match = VERDICT.search(guide)
    return match.group(1).capitalize() if match else None


def sentences(guide: str) -> list[str]:
    """Split a guide into lines and sentences. Good enough for keyword checks."""
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", guide) if part.strip()]


def says_input_must_be_images(guide: str) -> bool:
    """Return True if a sentence says the input has to be images."""
    return any(IMAGES_ONLY.search(part) for part in sentences(guide))


def partly_or_images_only(guide: str) -> bool:
    """Pass if the verdict is Partly, or the guide clearly says the input must be images."""
    return verdict_of(guide) == "Partly" or says_input_must_be_images(guide)


def names_image_formats(guide: str) -> bool:
    """Pass if the guide names all three formats the code accepts."""
    return all(re.search(rf"\b{word}\b", guide, re.I) for word in ("png", "jpg", "jpeg"))


def never_points_table_parsing_at_a_pdf(guide: str) -> bool:
    """Fail on an instruction to give the tool a PDF directly, unless the sentence says to convert it or warns against it."""
    return not any(
        re.search(r"\bpdfs?\b", part, re.I)
        and re.search(r"\b(point|set|input|inputs|pass|give)\b", part, re.I)
        and not NEGATION.search(part)
        # "a directory containing images of your PDF pages" is the right advice, not a PDF as input.
        and not re.search(r"\bimages? of\b", part, re.I)
        for part in sentences(guide)
    )


def paddleocr_paths_in_the_right_config(guide: str) -> bool:
    """Fail if PaddleOCR is placed in table_parsing.yaml, or discussed without naming configs/ocr.yaml."""
    wrong_file = any(
        "table_parsing.yaml" in part and "paddleocr" in part.lower() and not NEGATION.search(part)
        for part in sentences(guide)
    )
    return not wrong_file and ("paddleocr" not in guide.lower() or "configs/ocr.yaml" in guide)


def saves_pages_as_image_files(guide: str) -> bool:
    """Pass if the guide says to save pages as files, because the table model opens images by path."""
    return any(SAVES_IMAGES.search(part) for part in sentences(guide))


def includes_the_model_download(guide: str) -> bool:
    """Pass if the guide says to download the model weights, because the config expects them on disk."""
    return any(MODEL_DOWNLOAD.search(part) for part in sentences(guide))


def trusts_that_the_cpu_requirements_exist(guide: str) -> bool:
    """Fail if the guide casts doubt on requirements-cpu.txt, which is in the repo."""
    return not any("requirements-cpu.txt" in part and DOUBT.search(part) for part in sentences(guide))


CASES = [
    GuideCase(
        problem="extract tables from scanned PDFs in Python",
        repo="opendatalab/pdf-extract-kit",
        description="A Comprehensive Toolkit for High-Quality PDF Content Extraction",
        facts=[
            Fact("verdict is Partly, or says the input must be images", partly_or_images_only),
            Fact("names the accepted input formats: png, jpg, jpeg", names_image_formats),
            Fact("doesn't say to point table parsing at a PDF", never_points_table_parsing_at_a_pdf),
            Fact("if PaddleOCR is discussed, names configs/ocr.yaml and not table_parsing.yaml", paddleocr_paths_in_the_right_config),
            Fact("says to save pages as image files first", saves_pages_as_image_files),
            Fact("includes the model download", includes_the_model_download),
            Fact("doesn't doubt requirements-cpu.txt", trusts_that_the_cpu_requirements_exist),
        ],
    ),
]


def evaluate(guide: str, case: GuideCase) -> list[tuple[str, bool]]:
    """Return each fact's name and whether the guide passes it."""
    return [(fact.name, fact.check(guide)) for fact in case.facts]


def report(guide: str, case: GuideCase) -> bool:
    """Print pass or fail per fact, and return True if all passed."""
    results = evaluate(guide, case)
    print(f"\n{case.repo}: {case.problem}\nverdict: {verdict_of(guide)}")
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    return all(passed for _, passed in results)


def generate(case: GuideCase, save_to: Path | None) -> str | None:
    """Write a new guide with the real pipeline and print its verification. Nothing goes in the database."""
    from sqlalchemy import select

    from app.db import Repo, SessionLocal
    from app.finder.chains import generate_problem_guide

    with SessionLocal() as db:
        repo = db.scalar(select(Repo).where(Repo.name == case.repo))
        if repo is None or repo.status != "ready":
            print(f"{case.repo} isn't indexed and ready, so skipping it. Add it with POST /api/repos first.")
            return None
        result = generate_problem_guide(case.problem, repo, case.description)
    check = result.verification
    print(f"\nverification: {check.status}, changed: {check.changed}, files checked: {len(check.files_checked)}")
    for change in check.changes:
        print(f"  - {change}")
    if save_to is not None:
        save_to.write_text(result.guide, encoding="utf-8")
    return result.guide


def main() -> int:
    """Run the checks. Exit with 0 if every fact passed, 1 if not, and 2 if there was nothing to check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--guide-file", type=Path, help="check this saved guide against the first case")
    source.add_argument("--generate", action="store_true", help="write new guides (real model calls)")
    parser.add_argument("--save", type=Path, help="with --generate, save the guide to this file")
    args = parser.parse_args()
    # Guides are written with arrows and other symbols, which some Windows consoles can't print.
    sys.stdout.reconfigure(encoding="utf-8")

    if args.guide_file:
        return 0 if report(args.guide_file.read_text(encoding="utf-8"), CASES[0]) else 1
    guides = [(case, generate(case, args.save)) for case in CASES]
    checked = [(case, guide) for case, guide in guides if guide is not None]
    if not checked:
        return 2
    return 0 if all([report(guide, case) for case, guide in checked]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
