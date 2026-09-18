from dataclasses import dataclass
from enum import StrEnum


class LLMTask(StrEnum):
    EXTRACT = "extract"
    ANALYZE = "analyze"
    QUERY = "query"


class Provider(StrEnum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None


class ProviderError(Exception):
    def __init__(self, status: int, message: str, retry_after: float | None = None) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.retry_after = retry_after

    @property
    def is_retryable(self) -> bool:
        return self.status == 429 or self.status >= 500


class ExtractionFailed(Exception):
    """Both providers returned output that failed schema validation."""


class QuotaExhausted(Exception):
    """Both providers rate-limited."""

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("quota exhausted")
        self.retry_after = retry_after


class NoProvider(Exception):
    """No provider configured or reachable."""
