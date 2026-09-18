import pytest

from app.llm.prompts import load_prompt, render


def test_load_and_render() -> None:
    text = load_prompt("repair")
    assert "{error}" in text
    out = render("repair", error="bad field", previous='{"a":1}')
    assert "bad field" in out and '{"a":1}' in out


def test_missing_prompt() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")
