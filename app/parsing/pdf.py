from dataclasses import dataclass
from typing import Any

import pymupdf as fitz

Block = tuple[float, float, float, float, str]
_Span = tuple[int, tuple[float, float, float, float], str]  # band, bbox, text


@dataclass
class PageText:
    number: int
    text: str
    char_count: int


def order_blocks(blocks: list[Block], page_width: float, bands: int = 2) -> list[str]:
    """Cluster blocks into `bands` column bands by x-midpoint, then read each band
    top-to-bottom. `bands=1` is plain reading order for single-column pages."""
    if not blocks:
        return []
    band_width = page_width / bands  # ponytail: ≤2 bands; add k-means if 3-column resumes appear
    keyed = sorted(
        blocks,
        key=lambda b: (min(int(((b[0] + b[2]) / 2) // band_width), bands - 1), b[1], b[0]),
    )
    return [b[4].strip() for b in keyed if b[4].strip()]


def _merge(group: list[_Span]) -> Block:
    boxes = [g[1] for g in group]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
        " ".join(g[2].strip() for g in group),
    )


def _spans(page: fitz.Page) -> list[list[dict[str, Any]]]:
    """Non-empty spans per line, left to right."""
    lines: list[list[dict[str, Any]]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type", 0) != 0:
            continue
        for line in block["lines"]:
            spans = sorted(
                (s for s in line["spans"] if s["text"].strip()), key=lambda s: s["bbox"][0]
            )
            if spans:
                lines.append(spans)
    return lines


def is_two_column(lines: list[list[dict[str, Any]]], page_width: float) -> bool:
    """True when text sits in two side-by-side bands: few spans straddle the midpoint and
    both halves carry real text. Single-column resumes have many long lines that cross it."""
    mid = page_width / 2
    left = right = straddle = 0
    for spans in lines:  # judge whole lines, not spans: a line split into runs is still one line
        x0 = min(s["bbox"][0] for s in spans)
        x1 = max(s["bbox"][2] for s in spans)
        if x0 < mid - page_width * 0.05 and x1 > mid + page_width * 0.05:
            straddle += 1
        elif x1 <= mid:
            left += 1
        elif x0 >= mid:
            right += 1
    total = left + right + straddle
    if total < 4:
        return False
    return straddle / total < 0.1 and min(left, right) / total > 0.2


def _line_blocks(page: fitz.Page, bands: int) -> list[Block]:
    """One Block per (line, column band). PyMuPDF merges same-baseline text across columns
    into a single block, so split at span level — only on pages that really have two
    columns (`bands=2`); otherwise every line is one block."""
    lines = _spans(page)
    width = page.rect.width
    band_width = width / bands
    out: list[Block] = []
    for spans in lines:
        group: list[_Span] = []
        for s in spans:
            x0, y0, x1, y1 = s["bbox"]
            band = min(int(((x0 + x1) / 2) // band_width), bands - 1)
            if group and band != group[-1][0]:
                out.append(_merge(group))
                group = []
            group.append((band, (x0, y0, x1, y1), s["text"]))
        if group:
            out.append(_merge(group))
    return out


def extract_pages(doc: fitz.Document) -> list[PageText]:
    pages: list[PageText] = []
    for i, page in enumerate(doc):
        bands = 2 if is_two_column(_spans(page), page.rect.width) else 1
        text = "\n".join(order_blocks(_line_blocks(page, bands), page.rect.width, bands))
        pages.append(PageText(i + 1, text, len(text)))
    return pages
