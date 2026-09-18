import fitz  # PyMuPDF


def make_pdf(pages: list[list[tuple[float, float, str]]]) -> bytes:
    doc = fitz.open()
    for items in pages:
        page = doc.new_page(width=595, height=842)
        for x, y, text in items:
            page.insert_text((x, y), text, fontsize=11)
    return doc.tobytes()


def make_blank_pdf(n: int) -> bytes:
    doc = fitz.open()
    for _ in range(n):
        doc.new_page()
    return doc.tobytes()
