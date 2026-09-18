import json
from pathlib import Path

import pytest

from app.parsing.pipeline import parse_pdf

FIX = Path("tests/fixtures")
LABELS = json.loads((FIX / "golden/labels.json").read_text())


@pytest.mark.parametrize("fname", sorted(LABELS))
def test_fixture_parses_with_expected_fields(fname: str) -> None:
    if fname == "10_jonas.pdf":
        pytest.skip("image-only fixture needs Tesseract; OCR path covered by unit tests")
    doc = parse_pdf((FIX / "resumes" / fname).read_bytes())
    lab = LABELS[fname]
    assert lab["email"] in doc.text
    for skill in lab["skills"]:
        assert skill in doc.text
    assert doc.extraction_method == "pymupdf"
    assert "skills" in doc.sections and "experience" in doc.sections


def test_two_column_fixture_keeps_columns_ordered() -> None:
    text = parse_pdf((FIX / "resumes" / "03_chen.pdf").read_bytes()).text
    assert text.index("SKILLS") < text.index("EXPERIENCE") < text.index("EDUCATION")
