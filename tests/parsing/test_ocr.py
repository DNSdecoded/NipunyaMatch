from typing import Any

from app.parsing.ocr import needs_ocr, ocr_page
from app.parsing.pdf import PageText
from app.parsing.validate import validate_pdf
from tests.parsing.helpers import make_blank_pdf


def test_needs_ocr_threshold() -> None:
    assert needs_ocr(PageText(1, "x" * 99, 99))
    assert not needs_ocr(PageText(1, "x" * 100, 100))


def test_ocr_page_rasterises_at_300dpi_and_calls_tesseract() -> None:
    seen: list[Any] = []

    def fake(img: Any) -> str:
        seen.append(img)
        return "OCR TEXT"

    doc = validate_pdf(make_blank_pdf(1))
    assert ocr_page(doc, 0, run_tesseract=fake) == "OCR TEXT"
    img = seen[0]
    # A4 at 300 DPI ≈ 2480 x 3508
    assert 2400 < img.width < 2560 and 3400 < img.height < 3600
