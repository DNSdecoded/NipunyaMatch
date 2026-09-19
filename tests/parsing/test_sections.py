from app.parsing.sections import tag_sections


def test_tags_known_headings_with_synonyms() -> None:
    text = "WORK EXPERIENCE\nAcme 2020-2022\nTechnical Skills\nPython, SQL\nEDUCATION\nB.Tech 2019"
    s = tag_sections(text)
    assert s["experience"].strip() == "Acme 2020-2022"
    assert s["skills"].strip() == "Python, SQL"
    assert s["education"].strip() == "B.Tech 2019"


def test_no_headings_gives_empty_dict() -> None:
    assert tag_sections("just a paragraph about me") == {}


def test_compound_headings_match_by_prefix() -> None:
    text = "SKILLS AND INTRESTS" + chr(10) + "C, Verilog" + chr(10) + "Education & Training" + chr(10) + "B.Tech"
    s = tag_sections(text)
    assert s["skills"].strip() == "C, Verilog" and s["education"].strip() == "B.Tech"
