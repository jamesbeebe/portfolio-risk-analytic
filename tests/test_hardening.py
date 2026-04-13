import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def _balanced_portfolio_payload() -> dict:
    """Build a valid balanced portfolio payload for hardening tests.

    Returns:
        A JSON-ready payload that satisfies the public-demo API limits.
    """

    return {
        "tickers": ["AAPL", "MSFT", "SPY", "GLD"],
        "weights": [0.25, 0.25, 0.30, 0.20],
        "start_date": "2021-01-01",
        "end_date": "2026-01-01",
        "confidence_level": 0.95,
        "simulations": 10000,
        "horizon_days": 1,
        "random_seed": 42,
    }


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


def test_max_ticker_count_enforced() -> None:
    # Verifies oversized portfolios are rejected so public-demo compute remains bounded at the API boundary.
    payload = _balanced_portfolio_payload()
    payload["tickers"] = [f"TICKER{i}" for i in range(11)]
    payload["weights"] = [1 / 11] * 11
    response = client.post("/analyze", json=payload)

    assert response.status_code == 422
    assert "maximum of 10 tickers" in response.json()["detail"].lower()


def test_invalid_simulation_count_rejected() -> None:
    # Verifies disallowed simulation sizes are rejected so expensive requests stay on approved demo tiers only.
    payload = _balanced_portfolio_payload()
    payload["simulations"] = 20000
    response = client.post("/analyze", json=payload)

    assert response.status_code == 422
    assert "must be one of" in response.json()["detail"].lower()


def test_invalid_horizon_days_rejected() -> None:
    # Verifies non-demo horizon lengths are rejected so the public app stays limited to a one-day risk horizon.
    payload = _balanced_portfolio_payload()
    payload["horizon_days"] = 5
    response = client.post("/analyze", json=payload)

    assert response.status_code == 422
    assert "horizon_days must be 1" in response.json()["detail"].lower()


def test_too_long_date_range_rejected() -> None:
    # Verifies overly long history windows are rejected so the demo avoids excessive historical data requests.
    payload = _balanced_portfolio_payload()
    payload["start_date"] = "2010-01-01"
    payload["end_date"] = "2026-01-01"
    response = client.post("/analyze", json=payload)

    assert response.status_code == 422
    assert "cannot exceed 10 years" in response.json()["detail"].lower()


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
