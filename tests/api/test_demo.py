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
