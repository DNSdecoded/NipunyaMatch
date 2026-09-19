import json
import shutil
from pathlib import Path

import pytest

from app.parsing.pipeline import parse_pdf

FIX = Path("tests/fixtures")
LABELS = json.loads((FIX / "golden/labels.json").read_text())
_HAS_TESSERACT = bool(
    shutil.which("tesseract") or Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists()
)


@pytest.mark.parametrize("fname", sorted(LABELS))
def test_fixture_parses_with_expected_fields(fname: str) -> None:
    if fname == "10_jonas.pdf" and not _HAS_TESSERACT:
        pytest.skip("image-only fixture needs the tesseract binary")
    doc = parse_pdf((FIX / "resumes" / fname).read_bytes())
    lab = LABELS[fname]
    if fname == "10_jonas.pdf":  # image-only: OCR'd; exact strings depend on Tesseract
        assert doc.extraction_method == "pymupdf+ocr" and doc.char_count > 200
        assert lab["name"].split()[0] in doc.text
        return
    assert lab["email"] in doc.text
    for skill in lab["skills"]:
        assert skill in doc.text
    assert doc.extraction_method == "pymupdf"
    assert "skills" in doc.sections and "experience" in doc.sections


def test_two_column_fixture_keeps_columns_ordered() -> None:
    text = parse_pdf((FIX / "resumes" / "03_chen.pdf").read_bytes()).text
    assert text.index("SKILLS") < text.index("EXPERIENCE") < text.index("EDUCATION")
