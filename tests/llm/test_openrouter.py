import json

import httpx
import pytest
import respx

from app.llm.openrouter import OpenRouterClient
from app.llm.types import ProviderError

BASE = "https://openrouter.ai/api/v1"
SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


@respx.mock
async def test_generate_success() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"x": 2}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )
    )
    r = await OpenRouterClient("k", BASE).generate("google/gemini-3.8-flash", "hi", SCHEMA)
    assert r.text == '{"x": 2}' and r.prompt_tokens == 5
    body = json.loads(route.calls[0].request.read())
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert route.calls[0].request.headers["authorization"] == "Bearer k"


@respx.mock
async def test_error_status() -> None:
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(429, text="slow down"))
    with pytest.raises(ProviderError) as e:
        await OpenRouterClient("k", BASE).generate("m", "hi", SCHEMA)
    assert e.value.status == 429
