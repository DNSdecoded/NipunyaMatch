import io
from collections.abc import Callable
from typing import Any

import pymupdf as fitz
from PIL import Image

from app.parsing.pdf import PageText

OCR_THRESHOLD = 100
DPI = 300


def _tesseract(img: Any) -> str:
    import pytesseract

    return str(pytesseract.image_to_string(img))


def needs_ocr(page: PageText) -> bool:
    return page.char_count < OCR_THRESHOLD


def ocr_page(
    doc: fitz.Document, page_number: int, run_tesseract: Callable[[Any], str] = _tesseract
) -> str:
    pix = doc[page_number].get_pixmap(dpi=DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return run_tesseract(img)
