"""End-to-end smoke tests for the live FastAPI backend.

These tests require the FastAPI backend to be running on localhost:8000.
Run with: pytest tests/test_e2e_smoke.py --tb=short
Skip if the server is not running.
"""

from __future__ import annotations

import pytest
import requests

API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 15


@pytest.fixture
def api_available() -> None:
    """Ensure the live API server is reachable before running smoke tests.

    Returns:
        None. The fixture only gates test execution.

    Raises:
        pytest.skip: If the FastAPI server is not reachable on localhost:8000.
    """

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        pytest.skip("API server not running")

    if response.status_code != 200:
        pytest.skip("API server not running")


def _balanced_portfolio_payload() -> dict:
    """Build the balanced ETF sample payload used across smoke tests.

    Returns:
        A JSON-ready dictionary matching the first sample portfolio.
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


def test_full_analyze_flow(api_available: None) -> None:
    # Verifies the live /analyze endpoint returns the core risk metrics the Streamlit UI depends on.
    response = requests.post(
        f"{API_BASE_URL}/analyze",
        json=_balanced_portfolio_payload(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    assert response.status_code == 200
    body = response.json()
    assert "var_95" in body
    assert "es_95" in body
    assert "var_99" in body
    assert "es_99" in body
    assert "annualized_volatility" in body
    assert "correlation" in body
    assert body["es_95"] >= body["var_95"]
    assert body["es_99"] >= body["var_99"]
    assert 0 < body["annualized_volatility"] < 1


def test_full_simulate_flow(api_available: None) -> None:
    # Verifies the live /simulate endpoint returns percentile and tail-distribution data used in the UI charts.
    response = requests.post(
        f"{API_BASE_URL}/simulate",
        json=_balanced_portfolio_payload(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    assert response.status_code == 200
    body = response.json()
    assert "percentiles" in body
    assert "p5" in body["percentiles"]
    assert "p95" in body["percentiles"]
    assert body["percentiles"]["p5"] < body["percentiles"]["p95"]
    assert body["worst_case"] < body["best_case"]


def test_sample_portfolios_usable(api_available: None) -> None:
    # Verifies the sample portfolio endpoint returns payloads that can be submitted directly to /analyze.
    sample_response = requests.get(
        f"{API_BASE_URL}/sample-portfolios",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert sample_response.status_code == 200

    sample_portfolio = sample_response.json()["portfolios"][0]
    analyze_response = requests.post(
        f"{API_BASE_URL}/analyze",
        json=sample_portfolio,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert analyze_response.status_code == 200


def test_reproducibility_across_http(api_available: None) -> None:
    # Verifies identical seeded requests over real HTTP produce stable VaR output across repeated backend calls.
    first_response = requests.post(
        f"{API_BASE_URL}/analyze",
        json=_balanced_portfolio_payload(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    second_response = requests.post(
        f"{API_BASE_URL}/analyze",
        json=_balanced_portfolio_payload(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["var_95"] == second_response.json()["var_95"]


def test_bad_payload_rejected(api_available: None) -> None:
    # Verifies the live API rejects invalid weights the same way the UI validation layer is expected to reject them.
    bad_payload = {
        "tickers": ["AAPL", "MSFT", "SPY"],
        "weights": [0.5, 0.5, 0.5],
        "start_date": "2021-01-01",
        "end_date": "2026-01-01",
        "confidence_level": 0.95,
        "simulations": 10000,
        "horizon_days": 1,
        "random_seed": 42,
    }
    response = requests.post(
        f"{API_BASE_URL}/analyze",
        json=bad_payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    assert response.status_code == 422
