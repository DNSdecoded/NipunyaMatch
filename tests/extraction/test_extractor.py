import json
from typing import Any

import httpx
import respx

from app.extraction.extractor import MAX_CHARS, extract_candidate
from app.llm.gateway import Gateway

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@respx.mock
async def test_extract_truncates_and_validates(gateway: Gateway) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok({"name": "A", "skills": ["Python", "Python ", "SQL"], "phone": "123"})
    )
    text = "x" * (MAX_CHARS + 500)
    ext, flags = await extract_candidate(gateway, text)
    assert ext.name == "A" and ext.skills == ["Python", "SQL"]
    assert "phone_uncertain" in flags
    body = json.loads(route.calls[0].request.read())
    prompt = body["contents"][0]["parts"][0]["text"]
    assert prompt.count("x") <= MAX_CHARS + 50  # allow for the letter x in instructions
    assert "never invent" in prompt.lower()
