from typing import Any

from app.parsing.pipeline import parse_pdf, parse_text
from tests.parsing.helpers import make_pdf


def test_parse_text_pdf() -> None:
    pdf = make_pdf([[(72, 72, "SKILLS"), (72, 100, "Python and Docker " * 8)]])
    doc = parse_pdf(pdf)
    assert doc.extraction_method == "pymupdf"
    assert "Python and Docker" in doc.text
    assert doc.char_count == len(doc.text)
    assert "skills" in doc.sections


def test_parse_scanned_pdf_uses_ocr_only_on_empty_pages() -> None:
    calls: list[Any] = []

    def fake(img: Any) -> str:
        calls.append(img)
        return "OCR RESULT " * 20

    pdf = make_pdf([[(72, 100, "real text " * 20)], []])  # page 2 blank
    doc = parse_pdf(pdf, run_tesseract=fake)
    assert doc.extraction_method == "pymupdf+ocr"
    assert len(calls) == 1
    assert "real text" in doc.text and "OCR RESULT" in doc.text


def test_parse_text_plain() -> None:
    doc = parse_text("  Skills\nPython  ")
    assert doc.extraction_method == "text" and doc.text == "Skills\nPython"
