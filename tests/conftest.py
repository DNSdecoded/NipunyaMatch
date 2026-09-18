from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db.models import Base
from app.llm.breaker import CircuitBreaker
from app.llm.gateway import Gateway
from app.llm.gemini import GeminiClient
from app.llm.openrouter import OpenRouterClient
from app.main import create_app

GEM_URL = "https://generativelanguage.googleapis.com/v1beta"
ORT_URL = "https://openrouter.ai/api/v1"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-gemini",
        openrouter_api_key="test-openrouter",
        database_url="sqlite:///:memory:",
    )


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def db(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as s:
        yield s


@pytest.fixture
def gateway(settings: Settings, session_factory: sessionmaker[Session]) -> Gateway:
    async def no_sleep(_: float) -> None:
        return None

    return Gateway(
        settings,
        session_factory,
        GeminiClient("k", GEM_URL),
        OpenRouterClient("k", ORT_URL),
        CircuitBreaker(clock=lambda: 0.0),
        sleep=no_sleep,
    )


@pytest.fixture
def client(
    settings: Settings, session_factory: sessionmaker[Session], gateway: Gateway
) -> Iterator[TestClient]:
    app = create_app(settings, session_factory, gateway)
    with TestClient(app) as c:
        yield c
