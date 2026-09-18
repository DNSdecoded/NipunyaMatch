import os
from typing import Any

import httpx

API = os.environ.get("API_BASE_URL", "http://localhost:8000")


class UiError(Exception):
    pass


def _check(r: httpx.Response) -> Any:
    if r.is_success:
        return r.json()
    try:
        message = r.json().get("message", r.text)
    except ValueError:
        message = r.text
    raise UiError(message)


def get(path: str, **params: Any) -> Any:
    return _check(httpx.get(f"{API}{path}", params=params, timeout=60))


def post(path: str, **kw: Any) -> Any:
    return _check(httpx.post(f"{API}{path}", timeout=300, **kw))


def health() -> dict[str, Any]:
    try:
        result: dict[str, Any] = get("/api/health")
        return result
    except Exception:
        return {"gemini_configured": False, "openrouter_configured": False, "breaker_state": "down"}
