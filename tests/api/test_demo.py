import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.llm.gateway import Gateway
from app.main import create_app
from app.scoring.engine import ScoringEngine

GEM = "https://generativelanguage.googleapis.com/v1beta"
JD_RESP = {"title": "Backend", "required_skills": ["Python"], "preferred_skills": [],
           "min_years": None, "education_level": None}


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@pytest.fixture
def demo_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"demo_mode": True})


@pytest.fixture
def demo_client(
    demo_settings: Settings, session_factory: sessionmaker[Session], gateway: Gateway,
    scoring_engine: ScoringEngine,
) -> Iterator[TestClient]:
    from app.api.batches import BATCHES

    BATCHES.clear()
    app = create_app(demo_settings, session_factory, gateway, scoring_engine)
    with TestClient(app) as c:
        yield c


def test_health_reports_demo_mode(client: TestClient, demo_client: TestClient) -> None:
    assert client.get("/api/health").json()["demo_mode"] is False
    assert demo_client.get("/api/health").json()["demo_mode"] is True


@respx.mock
def test_header_key_builds_per_visitor_gateway(client: TestClient) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=gem_ok(JD_RESP)
    )
    for _ in range(2):
        r = client.post("/api/jobs", data={"text": "jd A"}, headers={"X-Gemini-Key": "visitor-a"})
        assert r.status_code == 201, r.text
    client.post("/api/jobs", data={"text": "jd B"}, headers={"X-Gemini-Key": "visitor-b"})
    sent = [c.request.headers["x-goog-api-key"] for c in route.calls]
    assert sent == ["visitor-a", "visitor-b"]  # second A call was a cache hit
    assert len(client.app.state.gateways) == 2


@respx.mock
def test_no_header_uses_server_gateway_outside_demo(client: TestClient) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=gem_ok(JD_RESP)
    )
    assert client.post("/api/jobs", data={"text": "jd"}).status_code == 201
    assert route.calls[0].request.headers["x-goog-api-key"] == "k"  # conftest gateway key


def test_demo_without_key_is_401(demo_client: TestClient) -> None:
    r = demo_client.post("/api/jobs", data={"text": "jd"})
    assert r.status_code == 401 and r.json()["code"] == "NO_API_KEY"
    assert "aistudio.google.com" in r.json()["message"]
    r = demo_client.post("/api/jobs/1/query", json={"question": "top 5"})
    assert r.status_code == 401


def test_gateway_pool_is_bounded(client: TestClient) -> None:
    from app.api.deps import MAX_VISITOR_GATEWAYS

    with respx.mock:
        respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
            return_value=gem_ok(JD_RESP)
        )
        for i in range(MAX_VISITOR_GATEWAYS + 5):
            client.post("/api/jobs", data={"text": f"jd {i}"}, headers={"X-Gemini-Key": f"k{i}"})
    assert len(client.app.state.gateways) == MAX_VISITOR_GATEWAYS


def test_demo_forbids_deletes_and_rescore(demo_client: TestClient) -> None:
    for method, path in [("DELETE", "/api/jobs/1"), ("DELETE", "/api/candidates"),
                         ("DELETE", "/api/candidates/1"), ("POST", "/api/jobs/1/rescore")]:
        r = demo_client.request(method, path)
        assert r.status_code == 403 and r.json()["code"] == "DEMO_READ_ONLY", path


@respx.mock
def test_demo_upload_caps(demo_client: TestClient) -> None:
    respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(return_value=gem_ok(JD_RESP))
    h = {"X-Gemini-Key": "visitor"}
    job_id = demo_client.post("/api/jobs", data={"text": "jd"}, headers=h).json()["id"]
    many = [("files", (f"{i}.pdf", b"%PDF", "application/pdf")) for i in range(11)]
    r = demo_client.post(f"/api/jobs/{job_id}/resumes", files=many, headers=h)
    assert r.status_code == 400 and r.json()["code"] == "TOO_MANY_FILES"
    big = [("files", ("big.pdf", b"%PDF" + b"0" * (5 * 1024 * 1024), "application/pdf"))]
    r = demo_client.post(f"/api/jobs/{job_id}/resumes", files=big, headers=h)
    assert r.status_code == 413 and r.json()["code"] == "FILE_TOO_LARGE"
