from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import LLMCall
from app.llm.breaker import CircuitBreaker
from app.llm.cache import cache_key, get_cached, set_cached
from app.llm.gemini import GeminiClient
from app.llm.openrouter import OpenRouterClient
from app.llm.prompts import render
from app.llm.schema_utils import inline_refs
from app.llm.types import (
    ExtractionFailed,
    LLMResponse,
    LLMTask,
    NoProvider,
    Provider,
    ProviderError,
    QuotaExhausted,
)

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

BACKOFF = (1.0, 2.0, 4.0)
MAX_ATTEMPTS = 3
JITTER = 0.09


def schema_version(schema: type[BaseModel]) -> str:
    digest = hashlib.sha256(json.dumps(schema.model_json_schema(), sort_keys=True).encode())
    return f"{schema.__name__}:{digest.hexdigest()[:8]}"


class Gateway:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        gemini: GeminiClient | None,
        openrouter: OpenRouterClient | None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if gemini is None and openrouter is None:
            raise NoProvider("no LLM provider configured")
        self._settings = settings
        self._sessions = session_factory
        self._gemini = gemini
        self._openrouter = openrouter
        self.breaker = breaker or CircuitBreaker()
        self._sleep = sleep
        self.last_provider: Provider | None = None
        self.call_count = 0

    def _model_for(self, task: LLMTask) -> str:
        s = self._settings
        return {
            LLMTask.EXTRACT: s.gemini_model_extract,
            LLMTask.ANALYZE: s.gemini_model_analyze,
            LLMTask.QUERY: s.gemini_model_query,
        }[task]

    def _log(
        self,
        provider: Provider,
        model: str,
        task: LLMTask,
        attempt: int,
        started: float,
        status: str,
        resp: LLMResponse | None = None,
    ) -> None:
        self.call_count += 1
        with self._sessions() as s:
            s.add(
                LLMCall(
                    provider=provider.value,
                    model=model,
                    task=task.value,
                    attempt=attempt,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    status=status,
                    prompt_tokens=resp.prompt_tokens if resp else None,
                    completion_tokens=resp.completion_tokens if resp else None,
                )
            )
            s.commit()

    async def _gemini_with_retry(
        self, model: str, prompt: str, js: dict[str, Any], task: LLMTask
    ) -> LLMResponse | None:
        """None when Gemini is skipped (breaker) or exhausts retries."""
        if self._gemini is None or not self.breaker.allow():
            return None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                resp = await self._gemini.generate(model, prompt, js)
            except ProviderError as e:
                self._log(Provider.GEMINI, model, task, attempt, started, "error")
                self.breaker.record_failure()
                if not e.is_retryable or attempt == MAX_ATTEMPTS:
                    return None
                delay = e.retry_after or BACKOFF[attempt - 1] + random.random() * JITTER
                await self._sleep(delay)
                continue
            self._log(Provider.GEMINI, model, task, attempt, started, "ok", resp)
            self.breaker.record_success()
            self.last_provider = Provider.GEMINI
            return resp
        return None

    async def _openrouter_once(
        self, prompt: str, js: dict[str, Any], task: LLMTask
    ) -> LLMResponse:
        if self._openrouter is None:
            raise NoProvider("Gemini unavailable and OPENROUTER_API_KEY not set")
        model = self._settings.openrouter_model
        started = time.monotonic()
        try:
            resp = await self._openrouter.generate(model, prompt, js)
        except ProviderError as e:
            self._log(Provider.OPENROUTER, model, task, 1, started, "error")
            if e.status in (429, 402):  # 402 = OpenRouter account out of credits
                raise QuotaExhausted(e.retry_after) from e
            raise NoProvider(str(e)) from e
        self._log(Provider.OPENROUTER, model, task, 1, started, "ok", resp)
        self.last_provider = Provider.OPENROUTER
        return resp

    async def _raw(self, prompt: str, js: dict[str, Any], task: LLMTask) -> str:
        resp = await self._gemini_with_retry(self._model_for(task), prompt, js, task)
        if resp is None:
            resp = await self._openrouter_once(prompt, js, task)
        return resp.text

    async def complete(self, prompt: str, schema: type[T], task: LLMTask) -> T:
        key = cache_key(prompt, self._model_for(task), schema_version(schema))
        with self._sessions() as s:
            cached = get_cached(s, key)
        if cached is not None:
            return schema.model_validate_json(cached)

        js = inline_refs(schema.model_json_schema())
        text = await self._raw(prompt, js, task)
        try:
            parsed = schema.model_validate_json(text)
        except ValidationError as first:
            fix = render("repair", error=str(first)[:2000], previous=text[:4000])
            repair = prompt + "\n\n" + fix
            text = await self._raw(repair, js, task)
            try:
                parsed = schema.model_validate_json(text)
            except ValidationError as second:
                raise ExtractionFailed(str(second)[:500]) from second
        with self._sessions() as s:
            set_cached(s, key, parsed.model_dump_json())
        return parsed


def build_gateway(settings: Settings, session_factory: sessionmaker[Session]) -> Gateway:
    gemini = (
        GeminiClient(settings.gemini_api_key, settings.gemini_base_url)
        if settings.gemini_api_key
        else None
    )
    openrouter = (
        OpenRouterClient(settings.openrouter_api_key, settings.openrouter_base_url)
        if settings.openrouter_api_key
        else None
    )
    return Gateway(settings, session_factory, gemini, openrouter)
