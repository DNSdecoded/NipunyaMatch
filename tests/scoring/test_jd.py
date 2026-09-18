import json
from typing import Any

import httpx
import respx
from sqlalchemy.orm import Session

from app.llm.gateway import Gateway
from app.scoring.jd import JobRequirements, create_job, job_requirements, parse_jd

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@respx.mock
async def test_parse_jd_uses_analyze_model(gateway: Gateway) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=gem_ok({
            "title": "Backend Engineer", "required_skills": ["Python", "Docker"],
            "preferred_skills": ["AWS"], "min_years": 3, "education_level": "bachelor",
        })
    )
    r = await parse_jd(gateway, "We need a backend engineer...")
    assert r.title == "Backend Engineer" and r.min_years == 3 and route.call_count == 1


def test_create_job_and_roundtrip(db: Session) -> None:
    reqs = JobRequirements(
        title="T", required_skills=["Python", "Docker"], preferred_skills=["AWS"],
        min_years=3, education_level="bachelor",
    )
    job = create_job(db, reqs, "raw jd")
    assert job.id and job.min_years == 3 and job.education_level == "bachelor"
    back = job_requirements(job)
    assert sorted(back.required_skills) == ["Docker", "Python"]
    assert back.preferred_skills == ["AWS"]
