import json
from typing import Any

import httpx
import respx

from app.llm.gateway import Gateway
from app.query.classifier import classify
from app.query.schemas import Intent, QueryPlan

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@respx.mock
async def test_classify_uses_query_model_and_parses(gateway: Gateway) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        return_value=gem_ok({"intent": "filter_skill", "skill": "Docker", "has_skill": False})
    )
    plan = await classify(gateway, "Which candidates are missing Docker?")
    assert plan == QueryPlan(intent=Intent.FILTER_SKILL, skill="Docker", has_skill=False)


@respx.mock
async def test_unparseable_falls_back_to_semantic(gateway: Gateway) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        return_value=gem_ok({"intent": "nonsense"})
    )
    plan = await classify(gateway, "tell me something")
    assert plan.intent == Intent.SEMANTIC
