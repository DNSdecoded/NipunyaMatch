import pymupdf as fitz
import pytest

from app.parsing.validate import InvalidPDF, validate_pdf
from tests.parsing.helpers import make_blank_pdf, make_pdf


def test_valid_pdf_returns_doc() -> None:
    doc = validate_pdf(make_pdf([[(72, 72, "hello")]]))
    assert doc.page_count == 1


def test_not_a_pdf() -> None:
    with pytest.raises(InvalidPDF) as e:
        validate_pdf(b"hello world")
    assert e.value.code == "INVALID_PDF" and "not a PDF" in e.value.message


def test_too_many_pages() -> None:
    with pytest.raises(InvalidPDF) as e:
        validate_pdf(make_blank_pdf(20))
    assert "20 pages" in e.value.message


def test_too_large() -> None:
    with pytest.raises(InvalidPDF) as e:
        validate_pdf(b"%PDF-" + b"0" * (10 * 1024 * 1024))
    assert e.value.code == "FILE_TOO_LARGE"


def test_encrypted() -> None:
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="x")
    with pytest.raises(InvalidPDF) as e:
        validate_pdf(data)
    assert "encrypted" in e.value.message
