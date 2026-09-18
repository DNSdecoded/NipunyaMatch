from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.parsing.normalise import normalise
from app.parsing.ocr import _tesseract, needs_ocr, ocr_page
from app.parsing.pdf import extract_pages
from app.parsing.sections import tag_sections
from app.parsing.validate import validate_pdf


@dataclass
class ParsedDocument:
    text: str
    extraction_method: str
    char_count: int
    sections: dict[str, str]


def parse_pdf(data: bytes, run_tesseract: Callable[[Any], str] = _tesseract) -> ParsedDocument:
    doc = validate_pdf(data)
    pages = extract_pages(doc)
    used_ocr = False
    texts: list[str] = []
    for p in pages:
        if needs_ocr(p):
            used_ocr = True
            texts.append(ocr_page(doc, p.number - 1, run_tesseract))
        else:
            texts.append(p.text)
    text = normalise(texts)
    return ParsedDocument(
        text=text,
        extraction_method="pymupdf+ocr" if used_ocr else "pymupdf",
        char_count=len(text),
        sections=tag_sections(text),
    )


def parse_text(text: str) -> ParsedDocument:
    clean = normalise([text])
    return ParsedDocument(clean, "text", len(clean), tag_sections(clean))
