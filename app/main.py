import asyncio
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker

from app.api.admin import router as admin_router
from app.api.candidates import router as candidates_router
from app.api.errors import install_error_handlers
from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.config import Settings, get_settings
from app.db.session import init_db, make_engine, make_session_factory
from app.llm.types import ExtractionFailed, NoProvider, QuotaExhausted
from app.scoring.engine import ScoringEngine


def _err(status: int, code: str, message: str, retry_after: int | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"code": code, "message": message, "retry_after": retry_after},
        headers={"Retry-After": str(retry_after)} if retry_after else None,
    )


def create_app(
    settings: Settings,
    session_factory: sessionmaker[Session],
    gateway: Any = None,
    engine: ScoringEngine | None = None,
) -> FastAPI:
    logging.basicConfig(level=settings.log_level)
    app = FastAPI(title="NipunyaMatch", version="0.1.0")
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.gateway = gateway
    app.state.gateways = {}  # visitor key digest -> Gateway (see api/deps.get_gateway)
    if engine is None:
        from app.scoring.embeddings import LocalEmbedder
        from app.scoring.engine import build_engine

        engine = build_engine(LocalEmbedder())
    app.state.engine = engine
    app.state.embedder = engine.matcher.embedder
    app.state.llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm)
    install_error_handlers(app)

    @app.exception_handler(QuotaExhausted)
    async def _quota(_: Request, e: QuotaExhausted) -> JSONResponse:
        ra = int(e.retry_after) if e.retry_after else None
        msg = "Both LLM providers are rate-limited or out of credits. Try again later."
        if ra:
            msg += f" Retry in about {ra}s."
        return _err(429, "QUOTA_EXHAUSTED", msg, ra)

    @app.exception_handler(NoProvider)
    async def _noprov(_: Request, e: NoProvider) -> JSONResponse:
        return _err(503, "NO_PROVIDER", "No LLM provider is configured or reachable. Check .env.")

    @app.exception_handler(ExtractionFailed)
    async def _extract(_: Request, e: ExtractionFailed) -> JSONResponse:
        return _err(422, "EXTRACTION_FAILED", "The model returned unusable output. Try again.")

    app.include_router(health_router)
    app.include_router(jobs_router)
    app.include_router(candidates_router)
    app.include_router(admin_router)
    return app


def _default_app() -> FastAPI:
    from app.llm.gateway import build_gateway

    settings = get_settings()
    db_engine = make_engine(settings.database_url)
    init_db(db_engine)
    factory = make_session_factory(db_engine)
    return create_app(settings, factory, build_gateway(settings, factory))


def __getattr__(name: str) -> Any:
    # ponytail: lazy `app` so importing this module (tests) never needs GEMINI_API_KEY;
    # uvicorn `app.main:app` resolves it through this hook.
    if name == "app":
        return _default_app()
    raise AttributeError(name)
