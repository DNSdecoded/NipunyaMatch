import logging
from typing import Any

from fastapi import FastAPI
from sqlalchemy.orm import Session, sessionmaker

from app.api.errors import install_error_handlers
from app.api.health import router as health_router
from app.config import Settings, get_settings
from app.db.session import init_db, make_engine, make_session_factory


def create_app(
    settings: Settings,
    session_factory: sessionmaker[Session],
    gateway: Any = None,
) -> FastAPI:
    logging.basicConfig(level=settings.log_level)
    app = FastAPI(title="NipunyaMatch", version="0.1.0")
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.gateway = gateway
    install_error_handlers(app)
    app.include_router(health_router)
    return app


def _default_app() -> FastAPI:
    from app.llm.gateway import build_gateway

    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    return create_app(settings, factory, build_gateway(settings, factory))


def __getattr__(name: str) -> Any:
    # ponytail: lazy `app` so importing this module (tests) never needs GEMINI_API_KEY;
    # uvicorn `app.main:app` resolves it through this hook.
    if name == "app":
        return _default_app()
    raise AttributeError(name)
