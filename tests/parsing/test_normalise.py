from app.parsing.normalise import normalise


def test_ligatures_and_quotes() -> None:
    assert normalise(["ﬁnance “quoted” it’s"]) == 'finance "quoted" it\'s'


def test_repeated_header_footer_removed() -> None:
    pages = ["John Doe Resume\nSkills: Python\nPage 1", "John Doe Resume\nExperience\nPage 2"]
    out = normalise(pages)
    assert out.count("John Doe Resume") == 0
    assert "Page 1" not in out
    assert "Skills: Python" in out and "Experience" in out


def test_join_broken_lines_keep_bullets() -> None:
    out = normalise(["Built a scalable data\npipeline for analytics\n• Led team\n• Shipped API"])
    assert "Built a scalable data pipeline for analytics" in out
    assert "• Led team\n• Shipped API" in out


def test_collapse_whitespace() -> None:
    assert normalise(["a   b\n\n\n\nc"]) == "a b\n\nc"
