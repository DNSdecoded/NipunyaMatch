from app.llm.types import LLMTask, ProviderError


def test_task_values() -> None:
    assert LLMTask.EXTRACT == "extract"
    assert set(LLMTask) == {"extract", "analyze", "query"}


def test_provider_error_carries_status() -> None:
    e = ProviderError(429, "quota", retry_after=2.5)
    assert e.status == 429 and e.retry_after == 2.5
    assert e.is_retryable
    assert not ProviderError(400, "bad").is_retryable
