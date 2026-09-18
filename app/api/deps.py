import hashlib
from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.api.errors import ApiError
from app.llm.gateway import Gateway, build_gateway

MAX_VISITOR_GATEWAYS = 100  # ponytail: in-process pool, drop-oldest; enough for a demo


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_gateway(request: Request) -> Gateway:
    """Server gateway locally; per-visitor gateway (own key, no OpenRouter) when X-Gemini-Key
    is sent. In demo mode the header is mandatory so the server key is never spent."""
    state = request.app.state
    key = request.headers.get("x-gemini-key", "").strip()
    if not key:
        if state.settings.demo_mode:
            raise ApiError(
                401, "NO_API_KEY",
                "Enter your Gemini API key in the sidebar "
                "(free at https://aistudio.google.com/apikey).",
            )
        gateway: Gateway = state.gateway
        return gateway
    pool: dict[str, Gateway] = state.gateways
    digest = hashlib.sha256(key.encode()).hexdigest()
    gw = pool.get(digest)
    if gw is None:
        if len(pool) >= MAX_VISITOR_GATEWAYS:
            pool.pop(next(iter(pool)))  # dicts keep insertion order: drop the oldest
        visitor = state.settings.model_copy(
            update={"gemini_api_key": key, "openrouter_api_key": None}
        )
        gw = pool[digest] = build_gateway(visitor, state.session_factory)
    return gw


def forbid_in_demo(request: Request) -> None:
    if request.app.state.settings.demo_mode:
        raise ApiError(
            403, "DEMO_READ_ONLY", "Deletes and rescoring are disabled on the public demo."
        )
