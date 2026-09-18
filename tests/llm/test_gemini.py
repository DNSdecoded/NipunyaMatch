import json

import httpx
import pytest
import respx

from app.llm.gemini import GeminiClient
from app.llm.types import ProviderError

BASE = "https://generativelanguage.googleapis.com/v1beta"
SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


@respx.mock
async def test_generate_success() -> None:
    route = respx.post(f"{BASE}/models/gemini-3.8-flash:generateContent").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"x": 1}'}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3},
            },
        )
    )
    client = GeminiClient(api_key="k", base_url=BASE)
    r = await client.generate("gemini-3.8-flash", "hi", SCHEMA)
    assert r.text == '{"x": 1}' and r.prompt_tokens == 10 and r.completion_tokens == 3
    sent = route.calls[0].request
    assert sent.headers["x-goog-api-key"] == "k"
    body = json.loads(sent.read())
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseSchema"]["properties"]["x"]["type"] == "integer"


@respx.mock
async def test_429_raises_retryable_with_retry_after() -> None:
    respx.post(f"{BASE}/models/m:generateContent").mock(
        return_value=httpx.Response(
            429, headers={"retry-after": "7"}, json={"error": {"message": "RESOURCE_EXHAUSTED"}}
        )
    )
    client = GeminiClient(api_key="k", base_url=BASE)
    with pytest.raises(ProviderError) as e:
        await client.generate("m", "hi", SCHEMA)
    assert e.value.status == 429 and e.value.retry_after == 7.0 and e.value.is_retryable


@respx.mock
async def test_500_raises_retryable() -> None:
    respx.post(f"{BASE}/models/m:generateContent").mock(
        return_value=httpx.Response(500, text="boom")
    )
    client = GeminiClient(api_key="k", base_url=BASE)
    with pytest.raises(ProviderError) as e:
        await client.generate("m", "hi", SCHEMA)
    assert e.value.status == 500


def test_strip_keeps_field_named_title() -> None:
    from pydantic import BaseModel

    from app.llm.gemini import _strip_for_gemini
    from app.llm.schema_utils import inline_refs

    class M(BaseModel):
        title: str
        default: int = 0

    out = _strip_for_gemini(inline_refs(M.model_json_schema()))
    assert set(out["properties"]) == {"title", "default"}
    assert "title" not in out  # schema-level keyword still dropped
    assert "default" not in out["properties"]["default"]


@respx.mock
async def test_429_retry_delay_parsed_from_body() -> None:
    respx.post(f"{BASE}/models/m:generateContent").mock(
        return_value=httpx.Response(
            429, json={"error": {"message": "Quota exceeded. Please retry in 43.7s."}}
        )
    )
    with pytest.raises(ProviderError) as e:
        await GeminiClient(api_key="k", base_url=BASE).generate("m", "hi", SCHEMA)
    assert e.value.retry_after == 44.7
