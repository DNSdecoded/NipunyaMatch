from collections.abc import Iterator

import httpx
import pytest
import respx
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import LLMCall
from app.llm.breaker import CircuitBreaker
from app.llm.gateway import Gateway
from app.llm.gemini import GeminiClient
from app.llm.openrouter import OpenRouterClient
from app.llm.types import ExtractionFailed, LLMTask, Provider, QuotaExhausted

GEM = "https://generativelanguage.googleapis.com/v1beta"
ORT = "https://openrouter.ai/api/v1"


class Out(BaseModel):
    x: int


def gem_ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


def or_ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def gw(settings: Settings, session_factory: sessionmaker[Session], sleeps: list[float]) -> Gateway:
    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    return Gateway(
        settings,
        session_factory,
        GeminiClient("k", GEM),
        OpenRouterClient("k", ORT),
        CircuitBreaker(threshold=5, cooldown=60, clock=lambda: 0.0),
        sleep=fake_sleep,
    )


@pytest.fixture(autouse=True)
def _mock() -> Iterator[None]:
    with respx.mock:
        yield


async def test_gemini_success_and_cache_hit(gw: Gateway, db: Session) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok('{"x": 1}')
    )
    r1 = await gw.complete("p", Out, LLMTask.EXTRACT)
    r2 = await gw.complete("p", Out, LLMTask.EXTRACT)
    assert r1 == r2 == Out(x=1)
    assert route.call_count == 1
    assert gw.last_provider == Provider.GEMINI
    calls = db.scalars(select(LLMCall)).all()
    assert len(calls) == 1 and calls[0].status == "ok" and calls[0].task == "extract"


async def test_retries_then_falls_back(gw: Gateway, db: Session, sleeps: list[float]) -> None:
    gem = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=httpx.Response(429, json={})
    )
    ort = respx.post(f"{ORT}/chat/completions").mock(return_value=or_ok('{"x": 9}'))
    r = await gw.complete("p", Out, LLMTask.ANALYZE)
    assert r == Out(x=9)
    assert gem.call_count == 3 and ort.call_count == 1
    assert [round(s) for s in sleeps] == [1, 2]
    assert all(1.0 <= sleeps[0] < 1.1 for _ in [0]) and 2.0 <= sleeps[1] < 2.1
    assert gw.last_provider == Provider.OPENROUTER
    statuses = [c.status for c in db.scalars(select(LLMCall)).all()]
    assert statuses == ["error", "error", "error", "ok"]


async def test_retry_after_header_honoured(gw: Gateway, sleeps: list[float]) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        side_effect=[httpx.Response(503, headers={"retry-after": "3"}), gem_ok('{"x": 4}')]
    )
    assert await gw.complete("p", Out, LLMTask.QUERY) == Out(x=4)
    assert sleeps == [3.0]


async def test_breaker_open_skips_gemini(gw: Gateway) -> None:
    gem = respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent")
    respx.post(f"{ORT}/chat/completions").mock(return_value=or_ok('{"x": 5}'))
    for _ in range(5):
        gw.breaker.record_failure()
    assert gw.breaker.state == "open"
    assert await gw.complete("p", Out, LLMTask.EXTRACT) == Out(x=5)
    assert gem.call_count == 0


async def test_bad_json_repaired_once(gw: Gateway) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        side_effect=[gem_ok('{"x": "not int"}'), gem_ok('{"x": 7}')]
    )
    assert await gw.complete("p", Out, LLMTask.EXTRACT) == Out(x=7)
    assert route.call_count == 2
    second = route.calls[1].request.read().decode()
    assert "did not match the required JSON schema" in second


async def test_repair_fails_raises_extraction_failed(gw: Gateway) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok("not json at all")
    )
    with pytest.raises(ExtractionFailed):
        await gw.complete("p", Out, LLMTask.EXTRACT)


async def test_both_rate_limited_raises_quota(gw: Gateway) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=httpx.Response(429)
    )
    respx.post(f"{ORT}/chat/completions").mock(return_value=httpx.Response(429))
    with pytest.raises(QuotaExhausted):
        await gw.complete("p", Out, LLMTask.EXTRACT)


async def test_no_openrouter_still_works_on_gemini(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok('{"x": 1}')
    )
    g = Gateway(settings, session_factory, GeminiClient("k", GEM), None)
    assert await g.complete("p", Out, LLMTask.EXTRACT) == Out(x=1)


async def test_fresh_bypasses_cache(gw: Gateway) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        side_effect=[gem_ok('{"x": 1}'), gem_ok('{"x": 2}')]
    )
    assert await gw.complete("p", Out, LLMTask.EXTRACT) == Out(x=1)
    assert await gw.complete("p", Out, LLMTask.EXTRACT, fresh=True) == Out(x=2)
    assert await gw.complete("p", Out, LLMTask.EXTRACT) == Out(x=2)  # fresh result re-cached
    assert route.call_count == 2
