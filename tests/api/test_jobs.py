import json
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from tests.parsing.helpers import make_pdf

GEM = "https://generativelanguage.googleapis.com/v1beta"
JD_RESP = {"title": "Backend", "required_skills": ["Python", "Docker"], "preferred_skills": [],
           "min_years": 3, "education_level": "bachelor"}
EXT_RESP = {"name": "Alice Smith", "email": "alice@example.com", "skills": ["Python", "FastAPI"],
            "total_years_experience": 4, "education": [{"institution": "X", "degree": "B.Tech"}]}
AN_RESP = {"llm_fit_score": 80,
           "skill_matches": [{"skill": "Python", "present": True, "evidence": "Python", "required": True},
                             {"skill": "Docker", "present": False, "evidence": None, "required": True}],
           "strengths": ["s"], "weaknesses": ["w"], "summary": "ok",
           "interview_questions": ["a", "b", "c"], "reasoning": "r"}


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


def _resume_pdf() -> bytes:
    return make_pdf([[(72, 72, "Alice Smith alice@example.com"),
                      (72, 100, "Skills: Python, FastAPI. Built services in Python. " * 3),
                      (72, 130, "B.Tech 2019")]])


@respx.mock
def test_full_flow(client: TestClient) -> None:
    respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        side_effect=[gem_ok(JD_RESP), gem_ok(AN_RESP)]
    )
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok(EXT_RESP)
    )
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        side_effect=[gem_ok({"intent": "top_n", "n": 5}), gem_ok({"answer": "Alice leads."})]
    )

    r = client.post("/api/jobs", data={"text": "We need Python and Docker, 3+ years, bachelor's."})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["required_skills"] == ["Python", "Docker"]
    job_id = job["id"]
    assert [j["id"] for j in client.get("/api/jobs").json()] == [job_id]

    r = client.post(f"/api/jobs/{job_id}/resumes",
                    files=[("files", ("alice.pdf", _resume_pdf(), "application/pdf")),
                           ("files", ("bad.pdf", b"not a pdf", "application/pdf"))])
    assert r.status_code == 202
    batch_id = r.json()["batch_id"]

    b = client.get(f"/api/batches/{batch_id}").json()
    assert b["total"] == 2 and b["done"] == 2
    by_name = {f["filename"]: f for f in b["files"]}
    assert by_name["alice.pdf"]["status"] == "done"
    assert by_name["alice.pdf"]["extraction_method"] == "pymupdf"
    assert by_name["bad.pdf"]["status"] == "error" and "not a PDF" in by_name["bad.pdf"]["error"]

    rows = client.get(f"/api/jobs/{job_id}/candidates").json()
    assert len(rows) == 1 and rows[0]["name"] == "Alice Smith"
    assert rows[0]["matched"] == 1 and rows[0]["missing"] == 1
    cid = rows[0]["candidate_id"]

    d = client.get(f"/api/candidates/{cid}").json()
    assert d["components"]["required_skill_coverage"] == 50.0
    assert d["interview_questions"] == ["a", "b", "c"]

    assert client.get(f"/api/jobs/{job_id}/candidates?min_score=99").json() == []
    assert client.get(f"/api/jobs/{job_id}/candidates?skill=Python").json()[0]["name"] == "Alice Smith"

    q = client.post(f"/api/jobs/{job_id}/query", json={"question": "Top candidates?"}).json()
    assert q["answer"] == "Alice leads." and q["intent"] == "top_n"
    assert q["sources"][0]["candidate_id"] == cid and q["provider_used"] == "gemini"

    r = client.post(f"/api/jobs/{job_id}/rescore")
    assert r.status_code == 200 and r.json()["rescored"] == 1

    respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=gem_ok({**AN_RESP, "llm_fit_score": 40})
    )
    r = client.post(f"/api/candidates/{cid}/reanalyze?job_id={job_id}")
    assert r.status_code == 200 and r.json()["components"]["llm_fit_score"] == 40.0
    assert client.post(f"/api/candidates/999/reanalyze?job_id={job_id}").status_code == 404

    r = client.get(f"/api/jobs/{job_id}/export?format=csv")
    assert r.status_code == 200 and "Alice Smith" in r.text
    assert r.headers["content-type"].startswith("text/csv")
    r = client.get(f"/api/jobs/{job_id}/export?format=xlsx")
    assert r.headers["content-type"].startswith("application/vnd.openxmlformats")


def test_list_jobs_empty(client: TestClient) -> None:
    assert client.get("/api/jobs").json() == []


def test_job_not_found(client: TestClient) -> None:
    r = client.get("/api/jobs/999")
    assert r.status_code == 404 and r.json()["code"] == "NOT_FOUND"


def test_too_many_files(client: TestClient) -> None:
    files = [("files", (f"{i}.pdf", b"%PDF", "application/pdf")) for i in range(51)]
    r = client.post("/api/jobs/1/resumes", files=files)
    assert r.status_code == 400 and r.json()["code"] == "TOO_MANY_FILES"


@respx.mock
def test_query_quota_exhausted_maps_429(client: TestClient) -> None:
    respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(return_value=gem_ok(JD_RESP))
    job_id = client.post("/api/jobs", data={"text": "jd"}).json()["id"]
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        return_value=httpx.Response(429))
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(429, headers={"retry-after": "30"}))
    r = client.post(f"/api/jobs/{job_id}/query", json={"question": "top 5"})
    assert r.status_code == 429 and r.json()["code"] == "QUOTA_EXHAUSTED"
    assert r.json()["retry_after"] == 30
