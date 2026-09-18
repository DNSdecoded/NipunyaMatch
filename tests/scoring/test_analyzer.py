import json
from typing import Any

import httpx
import respx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Analysis
from app.extraction.persist import persist_candidate
from app.extraction.schemas import CandidateExtraction, Education
from app.llm.gateway import Gateway
from app.parsing.pipeline import ParsedDocument
from app.scoring.analyzer import analyze_candidate, rescore_candidate, scoring_view
from app.scoring.engine import ScoringEngine, load_config
from app.scoring.jd import JobRequirements, create_job
from app.scoring.skills import SkillMatcher
from tests.scoring.fakes import FakeEmbedder

GEM = "https://generativelanguage.googleapis.com/v1beta"
RESUME = "Alice Smith. Built FastAPI services in Python on PostgreSQL. B.Tech 2019."


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


def test_scoring_view_strips_pii() -> None:
    v = scoring_view({"name": "A", "email": "a@x.com", "phone": "1", "location": "L",
                      "linkedin_url": "u", "skills": ["Python"], "experience": [],
                      "education": [], "projects": [], "certifications": [],
                      "total_years_experience": 4})
    assert set(v) == {"skills", "experience", "education", "projects", "certifications",
                      "total_years_experience"}


@respx.mock
async def test_analyze_persists_and_rescore_reuses_llm_fit(gateway: Gateway, db: Session) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=gem_ok({
            "llm_fit_score": 78,
            "skill_matches": [
                {"skill": "Python", "present": True, "evidence": "services in Python", "required": True},
                {"skill": "Docker", "present": True, "evidence": "deployed with Docker", "required": True},
            ],
            "strengths": ["APIs"], "weaknesses": ["No Docker"], "summary": "ok",
            "interview_questions": ["a", "b", "c"], "reasoning": "r",
        })
    )
    engine = ScoringEngine(load_config(), SkillMatcher({}, FakeEmbedder({})))
    job = create_job(db, JobRequirements(title="t", required_skills=["Python", "Docker"],
                                         min_years=3, education_level="bachelor"), "jd")
    ext = CandidateExtraction(name="Alice", skills=["Python", "FastAPI"], total_years_experience=4,
                              education=[Education(institution="X", degree="B.Tech")])
    cand = persist_candidate(db, ParsedDocument(RESUME, "pymupdf", len(RESUME), {}), ext, [], "h")

    a = await analyze_candidate(gateway, db, engine, cand, job)
    prompt = json.loads(route.calls[0].request.read())["contents"][0]["parts"][0]["text"]
    assert "Python" in prompt
    assert '"name"' not in prompt and '"email"' not in prompt
    assert a.llm_fit_score == 78 and a.required_skill_coverage == 50.0
    assert a.final_score == round(0.35 * 50 + 0.15 * 100 + 20 + 10 + 0.2 * 78)
    rows = {m.skill.canonical_name: m.present for m in a.skill_matches}
    assert rows == {"Python": True, "Docker": False}  # Docker evidence not in resume

    a2 = await analyze_candidate(gateway, db, engine, cand, job)  # upsert, cached LLM
    assert a2.id == a.id and route.call_count == 1
    assert db.scalar(select(func.count()).select_from(Analysis)) == 1

    engine.config.weights["llm_fit_score"] = 0.0
    r = rescore_candidate(db, engine, cand, job)
    assert r.id == a.id and r.llm_fit_score == 78
    assert r.final_score == round(0.35 * 50 + 0.15 * 100 + 20 + 10)
