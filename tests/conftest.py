import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-gemini",
        openrouter_api_key="test-openrouter",
        database_url="sqlite:///:memory:",
    )
