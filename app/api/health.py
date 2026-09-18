from fastapi import APIRouter, Request

from app.api.errors import ApiError

router = APIRouter()


@router.get("/api/health")
async def health(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    gateway = request.app.state.gateway
    return {
        "gemini_configured": bool(settings.gemini_api_key),
        "openrouter_configured": bool(settings.openrouter_api_key),
        "breaker_state": gateway.breaker.state if gateway else "closed",
        "demo_mode": settings.demo_mode,
    }


@router.get("/api/_error_probe", include_in_schema=False)
async def error_probe() -> None:
    raise ApiError(418, "PROBE", "probe", retry_after=5)
