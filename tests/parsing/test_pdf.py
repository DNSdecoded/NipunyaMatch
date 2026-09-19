from app.parsing.pdf import extract_pages, order_blocks
from app.parsing.validate import validate_pdf
from tests.parsing.helpers import make_pdf


def test_order_blocks_two_columns() -> None:
    blocks = [
        (300, 100, 500, 120, "R1"),
        (50, 100, 250, 120, "L1"),
        (50, 200, 250, 220, "L2"),
        (300, 200, 500, 220, "R2"),
    ]
    assert order_blocks(blocks, page_width=595) == ["L1", "L2", "R1", "R2"]


def test_order_blocks_single_column_by_y() -> None:
    blocks = [(50, 300, 500, 320, "B"), (50, 100, 500, 120, "A")]
    assert order_blocks(blocks, page_width=595) == ["A", "B"]


def test_extract_pages_two_column_pdf() -> None:
    pdf = make_pdf([[(72, 100, "Left top"), (350, 100, "Right top"),
                     (72, 200, "Left bottom"), (350, 200, "Right bottom")]])
    pages = extract_pages(validate_pdf(pdf))
    assert len(pages) == 1
    t = pages[0].text
    order = [t.index(x) for x in ("Left top", "Left bottom", "Right top", "Right bottom")]
    assert order == sorted(order)
    assert pages[0].char_count == len(t)


def test_single_column_long_lines_are_not_split() -> None:
    # Lines that cross the page midpoint must stay intact; this is a one-column resume.
    long = "AI engineer specializing in generative AI, agentic RAG systems and evaluation pipelines"
    pdf = make_pdf([[(40, 80, "Jane Doe"), (40, 100, long), (40, 120, "Summary"),
                     (40, 140, long + " again"), (40, 160, "Skills: Python, SQL, PyTorch " * 2),
                     (40, 180, long), (40, 200, long), (40, 220, long), (40, 240, long)]])
    text = extract_pages(validate_pdf(pdf))[0].text
    assert text.startswith("Jane Doe")
    assert long in text and long + " again" in text
