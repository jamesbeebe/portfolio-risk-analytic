import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_and_enable_rate_limiter() -> None:
    """Enable and reset the shared limiter for isolated hardening tests.

    Returns:
        None. The fixture ensures limiter state does not leak between tests.
    """

    previous_enabled = app.state.limiter.enabled
    app.state.limiter.enabled = True

    storage = getattr(app.state.limiter, "_storage", None)
    if storage is not None and hasattr(storage, "reset"):
        storage.reset()

    yield

    if storage is not None and hasattr(storage, "reset"):
        storage.reset()
    app.state.limiter.enabled = previous_enabled


def test_rate_limited_endpoint_allows_normal_request() -> None:
    # Verifies a normal request still succeeds when rate limiting is enabled for public-demo traffic.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rate_limit_returns_professional_429_response() -> None:
    # Verifies repeated requests eventually return a 429 response with a readable error body and retry guidance.
    last_response = None
    for _ in range(31):
        last_response = client.get("/")

    assert last_response is not None
    assert last_response.status_code == 429
    body = last_response.json()
    assert body["error"] == "rate_limit_exceeded"
    assert "Please wait" in body["detail"]
    assert last_response.headers["Retry-After"] == "60"
