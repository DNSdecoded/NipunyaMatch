from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    gemini_model_extract: str = "gemini-3.5-flash-lite"
    gemini_model_analyze: str = "gemini-3.8-flash"
    gemini_model_query: str = "gemini-3.5-flash"
    openrouter_model: str = "google/gemini-3.8-flash"
    database_url: str = "sqlite:///./data/recruiter.db"
    embedding_backend: Literal["local", "gemini"] = "local"
    max_concurrent_llm: int = 4
    log_level: str = "INFO"
    demo_mode: bool = False  # public demo: BYO key, no deletes, small upload caps
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @field_validator("gemini_api_key")
    @classmethod
    def _require_gemini_key(cls, v: str | None) -> str:
        if not v:
            raise ValueError(
                "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
                "and put it in .env (see .env.example)."
            )
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
