import json
from typing import Any

import httpx
import respx
from sqlalchemy.orm import Session

from app.llm.gateway import Gateway
from app.query.service import answer_question
from tests.query.seed import seed
from tests.scoring.fakes import FakeEmbedder

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@respx.mock
async def test_explain_ranking_end_to_end(gateway: Gateway, db: Session) -> None:
    job_id = seed(db)
    route = respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        side_effect=[
            gem_ok({"intent": "explain_ranking", "candidate_a": "Alice", "candidate_b": "Bob"}),
            gem_ok({"answer": "Alice ranks above Bob mainly on required-skill coverage, 100 vs 33."}),
        ]
    )
    r = await answer_question(gateway, db, job_id, "Why is Alice above Bob?", FakeEmbedder({}))
    assert r.intent == "explain_ranking"
    assert [s.name for s in r.sources] == ["Alice Smith", "Bob Jones"]
    assert r.provider_used == "gemini" and r.latency_ms >= 0
    synth_prompt = json.loads(route.calls[1].request.read())["contents"][0]["parts"][0]["text"]
    assert '"score_gap": 21' in synth_prompt
    assert "answer only from the facts" in synth_prompt.lower()


@respx.mock
async def test_empty_result_skips_synthesis(gateway: Gateway, db: Session) -> None:
    job_id = seed(db)
    route = respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        return_value=gem_ok({"intent": "filter_skill", "skill": "COBOL", "has_skill": True})
    )
    r = await answer_question(gateway, db, job_id, "Who knows COBOL?", FakeEmbedder({}))
    assert r.sources == [] and "No candidates" in r.answer and route.call_count == 1
