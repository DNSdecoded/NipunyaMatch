import io
from collections.abc import Callable
from typing import Any

import pymupdf as fitz
from PIL import Image

from app.parsing.pdf import PageText

OCR_THRESHOLD = 100
DPI = 300


_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _tesseract(img: Any) -> str:
    import shutil
    from pathlib import Path

    import pytesseract

    if shutil.which("tesseract") is None and Path(_WIN_TESSERACT).exists():
        pytesseract.pytesseract.tesseract_cmd = _WIN_TESSERACT  # winget install, not on PATH
    return str(pytesseract.image_to_string(img))


def needs_ocr(page: PageText) -> bool:
    return page.char_count < OCR_THRESHOLD


def ocr_page(
    doc: fitz.Document, page_number: int, run_tesseract: Callable[[Any], str] = _tesseract
) -> str:
    pix = doc[page_number].get_pixmap(dpi=DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return run_tesseract(img)
