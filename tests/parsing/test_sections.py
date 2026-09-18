from app.parsing.sections import tag_sections


def test_tags_known_headings_with_synonyms() -> None:
    text = "WORK EXPERIENCE\nAcme 2020-2022\nTechnical Skills\nPython, SQL\nEDUCATION\nB.Tech 2019"
    s = tag_sections(text)
    assert s["experience"].strip() == "Acme 2020-2022"
    assert s["skills"].strip() == "Python, SQL"
    assert s["education"].strip() == "B.Tech 2019"


def test_no_headings_gives_empty_dict() -> None:
    assert tag_sections("just a paragraph about me") == {}
