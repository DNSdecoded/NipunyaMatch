import json
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from tests.parsing.helpers import make_pdf

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


def pdf(name: str, skills: str) -> bytes:
    return make_pdf([[(72, 72, f"{name} {name.lower()}@example.com"),
                      (72, 100, f"Skills: {skills}. Built systems using {skills}. " * 3),
                      (72, 130, "B.Tech 2018")]])


def analysis(fit: int, present: dict[str, bool]) -> dict[str, Any]:
    return {"llm_fit_score": fit,
            "skill_matches": [{"skill": k, "present": v, "evidence": k if v else None,
                               "required": True} for k, v in present.items()],
            "strengths": ["s"], "weaknesses": ["w"], "summary": f"fit {fit}",
            "interview_questions": ["a", "b", "c"], "reasoning": "r"}


def _route_by_prompt(prompt_key: str, responses: dict[str, dict[str, Any]]) -> Any:
    """Pick a canned response by a substring of the prompt so concurrency order doesn't matter."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())["contents"][0]["parts"][0]["text"]
        for needle, resp in responses.items():
            if needle in body:
                return gem_ok(resp)
        raise AssertionError(f"no canned {prompt_key} response for prompt: {body[:200]}")

    return handler


@respx.mock
def test_cold_start_to_grounded_answer(client: TestClient) -> None:
    respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        side_effect=_route_by_prompt("analyze", {
            "Job description:": {"title": "Backend", "required_skills": ["Python", "Docker"],
                                 "preferred_skills": [], "min_years": 2, "education_level": "bachelor"},
            "alice@example.com": analysis(90, {"Python": True, "Docker": True}),
            "bob@example.com": analysis(60, {"Python": True, "Docker": False}),
        })
    )
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        side_effect=_route_by_prompt("extract", {
            "alice@example.com": {"name": "Alice", "skills": ["Python", "Docker"],
                                  "total_years_experience": 5,
                                  "education": [{"institution": "X", "degree": "B.Tech"}]},
            "bob@example.com": {"name": "Bob", "skills": ["Python"], "total_years_experience": 3,
                                "education": [{"institution": "X", "degree": "B.Tech"}]},
        })
    )
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        side_effect=_route_by_prompt("query", {
            "Classify": {"intent": "explain_ranking", "candidate_a": "Alice", "candidate_b": "Bob"},
            "Facts (JSON)": {"answer": "Alice ranks above Bob on required-skill coverage: Docker present."},
        })
    )

    job_id = client.post("/api/jobs", data={"text": "Python, Docker, 2+ years"}).json()["id"]
    batch = client.post(f"/api/jobs/{job_id}/resumes", files=[
        ("files", ("alice.pdf", pdf("Alice", "Python, Docker"), "application/pdf")),
        ("files", ("bob.pdf", pdf("Bob", "Python"), "application/pdf")),
    ]).json()["batch_id"]
    assert client.get(f"/api/batches/{batch}").json()["done"] == 2

    rows = client.get(f"/api/jobs/{job_id}/candidates").json()
    assert [r["name"] for r in rows] == ["Alice", "Bob"]
    assert rows[0]["final_score"] > rows[1]["final_score"]

    q = client.post(f"/api/jobs/{job_id}/query", json={"question": "Why is Alice above Bob?"}).json()
    assert q["intent"] == "explain_ranking"
    assert [s["name"] for s in q["sources"]] == ["Alice", "Bob"]
    assert "Docker" in q["answer"]
