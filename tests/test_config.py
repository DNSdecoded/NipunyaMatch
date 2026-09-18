import pytest
from pydantic import ValidationError

from app.config import Settings


def test_missing_gemini_key_names_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None)
    assert "GEMINI_API_KEY" in str(exc.value)
    assert "aistudio.google.com" in str(exc.value)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    s = Settings(_env_file=None)
    assert s.gemini_model_extract == "gemini-3.5-flash-lite"
    assert s.gemini_model_analyze == "gemini-3.8-flash"
    assert s.gemini_model_query == "gemini-3.5-flash"
    assert s.openrouter_model == "google/gemini-3.8-flash"
    assert s.max_concurrent_llm == 4
    assert s.openrouter_api_key is None
