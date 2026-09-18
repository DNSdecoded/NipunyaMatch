from dataclasses import dataclass

import pymupdf as fitz

Block = tuple[float, float, float, float, str]
_Span = tuple[int, tuple[float, float, float, float], str]  # band, bbox, text


@dataclass
class PageText:
    number: int
    text: str
    char_count: int


def order_blocks(blocks: list[Block], page_width: float) -> list[str]:
    """Cluster blocks into column bands by x-midpoint, then read each band top-to-bottom."""
    if not blocks:
        return []
    band_width = page_width / 2  # ponytail: two bands; add k-means if 3-column resumes appear
    keyed = sorted(
        blocks,
        key=lambda b: (int(((b[0] + b[2]) / 2) // band_width), b[1], b[0]),
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


def _line_blocks(page: fitz.Page) -> list[Block]:
    """One Block per (line, column band). PyMuPDF merges same-baseline text across columns
    into a single block, so split at span level instead of trusting block bboxes."""
    band_width = page.rect.width / 2
    out: list[Block] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type", 0) != 0:
            continue
        for line in block["lines"]:
            spans = sorted(
                (s for s in line["spans"] if s["text"].strip()), key=lambda s: s["bbox"][0]
            )
            group: list[_Span] = []
            for s in spans:
                x0, y0, x1, y1 = s["bbox"]
                band = int(((x0 + x1) / 2) // band_width)
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
        text = "\n".join(order_blocks(_line_blocks(page), page.rect.width))
        pages.append(PageText(i + 1, text, len(text)))
    return pages
