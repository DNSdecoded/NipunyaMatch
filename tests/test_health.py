from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["gemini_configured"] is True
    assert body["openrouter_configured"] is True
    assert body["breaker_state"] == "closed"


def test_error_shape(client: TestClient) -> None:
    r = client.get("/api/_error_probe")
    assert r.status_code == 418
    assert r.json() == {"code": "PROBE", "message": "probe", "retry_after": 5}
    assert r.headers["retry-after"] == "5"
