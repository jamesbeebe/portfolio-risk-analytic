import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.db.database import Base, SessionLocal, engine
from app.db.models import AnalysisRun, SavedPortfolio
from app.models.results import RiskResults

client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_rate_limiter_for_api_tests() -> None:
    """Disable backend rate limiting so endpoint tests remain deterministic.

    Returns:
        None. The fixture temporarily disables the shared limiter for this file.
    """

    previous_enabled = app.state.limiter.enabled
    app.state.limiter.enabled = False
    yield
    app.state.limiter.enabled = previous_enabled


@pytest.fixture(autouse=True)
def reset_database_tables_for_api_tests() -> None:
    """Create required tables and clear persisted rows for deterministic API tests.

    Returns:
        None. The fixture prepares a clean local SQLite state around each test.
    """

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(AnalysisRun).delete()
        db.query(SavedPortfolio).delete()
        db.commit()
    finally:
        db.close()

    yield

    db = SessionLocal()
    try:
        db.query(AnalysisRun).delete()
        db.query(SavedPortfolio).delete()
        db.commit()
    finally:
        db.close()


def _balanced_portfolio_payload() -> dict:
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


def _tech_portfolio_payload() -> dict:
    return {
        "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "weights": [0.30, 0.30, 0.20, 0.20],
        "start_date": "2021-01-01",
        "end_date": "2026-01-01",
        "confidence_level": 0.99,
        "simulations": 10000,
        "horizon_days": 1,
        "random_seed": 42,
    }


def _sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAPL": [100.0, 101.5, 102.0, 101.0, 103.5, 104.0],
            "MSFT": [200.0, 201.0, 203.0, 202.5, 204.0, 205.5],
            "GOOGL": [150.0, 151.0, 152.5, 153.0, 154.0, 155.0],
            "NVDA": [500.0, 508.0, 512.0, 509.0, 520.0, 526.0],
            "SPY": [300.0, 301.0, 302.0, 301.5, 303.0, 304.5],
            "GLD": [180.0, 180.5, 181.0, 181.2, 181.8, 182.0],
        },
        index=pd.date_range("2021-01-04", periods=6, freq="B"),
    )


def _mock_risk_results() -> RiskResults:
    return RiskResults(
        tickers=["AAPL", "MSFT", "SPY", "GLD"],
        weights=[0.25, 0.25, 0.30, 0.20],
        mean_daily_return=0.0007,
        annualized_volatility=0.1243,
        var_95=0.0182,
        es_95=0.0241,
        var_99=0.0274,
        es_99=0.0347,
        correlation_matrix={
            "AAPL": {"AAPL": 1.0, "MSFT": 0.86, "SPY": 0.78, "GLD": 0.11},
            "MSFT": {"AAPL": 0.86, "MSFT": 1.0, "SPY": 0.76, "GLD": 0.09},
            "SPY": {"AAPL": 0.78, "MSFT": 0.76, "SPY": 1.0, "GLD": 0.04},
            "GLD": {"AAPL": 0.11, "MSFT": 0.09, "SPY": 0.04, "GLD": 1.0},
        },
        simulation_count=10000,
        horizon_days=1,
    )


def test_health_returns_200() -> None:
    # Verifies the health endpoint is reachable and reports an operational API status.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_returns_200() -> None:
    # Verifies the root endpoint responds successfully so users can discover the API entrypoint.
    response = client.get("/")
    assert response.status_code == 200


def test_health_allows_local_streamlit_origin() -> None:
    # Verifies local Streamlit development origins receive the expected CORS header for browser-based API calls.
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_sample_portfolios_returns_list() -> None:
    # Verifies the sample portfolio endpoint returns bundled example portfolios for demos and client bootstrapping.
    response = client.get("/sample-portfolios")
    body = response.json()
    assert response.status_code == 200
    assert isinstance(body["portfolios"], list)
    assert len(body["portfolios"]) > 0


def test_analyze_valid_portfolio(monkeypatch: object) -> None:
    # Verifies a valid analyze request returns the expected risk metrics and preserves the ES >= VaR guarantees.
    monkeypatch.setattr(
        "app.api.main.run_risk_pipeline", lambda portfolio: _mock_risk_results()
    )
    response = client.post("/analyze", json=_balanced_portfolio_payload())
    body = response.json()

    assert response.status_code == 200
    assert "var_95" in body
    assert "es_95" in body
    assert "var_99" in body
    assert "es_99" in body
    assert "annualized_volatility" in body
    assert body["es_95"] >= body["var_95"]
    assert body["es_99"] >= body["var_99"]


def test_analyze_deterministic_with_seed(monkeypatch: object) -> None:
    # Verifies repeated analyze calls with the same seed produce stable risk output for reproducible API behavior.
    monkeypatch.setattr(
        "app.api.main.run_risk_pipeline", lambda portfolio: _mock_risk_results()
    )
    first_response = client.post("/analyze", json=_balanced_portfolio_payload())
    second_response = client.post("/analyze", json=_balanced_portfolio_payload())

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["var_95"] == second_response.json()["var_95"]


def test_analyze_weights_dont_sum_to_one() -> None:
    # Verifies invalid portfolio weights are rejected before analytics run so the API enforces normalized allocations.
    payload = _balanced_portfolio_payload()
    payload["weights"] = [0.5, 0.5, 0.5, 0.5]
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_mismatched_tickers_and_weights() -> None:
    # Verifies ticker and weight count mismatches are rejected because each asset must have exactly one weight.
    payload = _balanced_portfolio_payload()
    payload["tickers"] = ["AAPL", "MSFT", "SPY"]
    payload["weights"] = [0.5, 0.5]
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_missing_tickers_field() -> None:
    # Verifies missing required fields are rejected so the API never runs the pipeline on incomplete requests.
    payload = _balanced_portfolio_payload()
    payload.pop("tickers")
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_confidence_level_out_of_range() -> None:
    # Verifies out-of-range confidence levels are blocked because the API contract defines valid risk-reporting bounds.
    payload = _balanced_portfolio_payload()
    payload["confidence_level"] = 0.50
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_rejects_more_than_ten_tickers() -> None:
    # Verifies the backend caps ticker count to protect expensive public-demo requests from oversized portfolios.
    payload = _balanced_portfolio_payload()
    payload["tickers"] = [f"TICKER{i}" for i in range(11)]
    payload["weights"] = [1 / 11] * 11
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_rejects_disallowed_simulation_count() -> None:
    # Verifies only approved simulation sizes are accepted so the backend can control compute cost predictably.
    payload = _balanced_portfolio_payload()
    payload["simulations"] = 20000
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_rejects_too_short_date_range() -> None:
    # Verifies very short date windows are blocked because the backend requires enough history for stable analysis.
    payload = _balanced_portfolio_payload()
    payload["start_date"] = "2025-01-01"
    payload["end_date"] = "2025-03-01"
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_rejects_too_long_date_range() -> None:
    # Verifies very long date windows are blocked to keep the public demo within bounded historical data limits.
    payload = _balanced_portfolio_payload()
    payload["start_date"] = "2010-01-01"
    payload["end_date"] = "2026-01-01"
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_rejects_non_demo_horizon() -> None:
    # Verifies the public demo enforces a fixed one-day horizon to limit complexity and backend compute scope.
    payload = _balanced_portfolio_payload()
    payload["horizon_days"] = 5
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_rejects_duplicate_tickers() -> None:
    # Verifies duplicate ticker symbols are blocked because repeated assets would distort portfolio composition and compute.
    payload = _balanced_portfolio_payload()
    payload["tickers"] = ["AAPL", "AAPL", "SPY", "GLD"]
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_simulate_returns_percentiles(monkeypatch: object) -> None:
    # Verifies the simulate endpoint returns percentile statistics and preserves distribution ordering from lower to upper tails.
    monkeypatch.setattr(
        "app.api.main.fetch_price_data",
        lambda **kwargs: _sample_prices()[kwargs["tickers"]],
    )
    response = client.post("/simulate", json=_tech_portfolio_payload())
    body = response.json()

    assert response.status_code == 200
    assert "percentiles" in body
    assert "p5" in body["percentiles"]
    assert "p95" in body["percentiles"]
    assert body["percentiles"]["p5"] < body["percentiles"]["p95"]


def test_save_portfolio_persists_record() -> None:
    # Verifies the save endpoint persists a named portfolio preset and returns the stored metadata cleanly.
    response = client.post(
        "/portfolios/save",
        json={
            "name": "Balanced ETF",
            "notes": "Core allocation",
            "portfolio": _balanced_portfolio_payload(),
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["name"] == "Balanced ETF"
    assert body["tickers"] == _balanced_portfolio_payload()["tickers"]
    assert body["weights"] == _balanced_portfolio_payload()["weights"]
    assert body["notes"] == "Core allocation"


def test_get_portfolios_returns_saved_rows() -> None:
    # Verifies the list endpoint returns saved portfolios in API-friendly form after persistence writes occur.
    save_response = client.post(
        "/portfolios/save",
        json={
            "name": "Balanced ETF",
            "portfolio": _balanced_portfolio_payload(),
        },
    )
    assert save_response.status_code == 200

    response = client.get("/portfolios")
    body = response.json()

    assert response.status_code == 200
    assert body["count"] == 1
    assert body["portfolios"][0]["name"] == "Balanced ETF"


def test_get_portfolio_by_id_returns_404_when_missing() -> None:
    # Verifies looking up a nonexistent saved portfolio returns a clean 404 error instead of crashing.
    response = client.get("/portfolios/9999")
    body = response.json()

    assert response.status_code == 404
    assert body["detail"] == "Portfolio with id 9999 not found"


def test_delete_portfolio_removes_record() -> None:
    # Verifies the delete endpoint removes saved portfolios and returns the expected success message.
    save_response = client.post(
        "/portfolios/save",
        json={
            "name": "Balanced ETF",
            "portfolio": _balanced_portfolio_payload(),
        },
    )
    portfolio_id = save_response.json()["id"]

    delete_response = client.delete(f"/portfolios/{portfolio_id}")
    list_response = client.get("/portfolios")

    assert delete_response.status_code == 200
    assert delete_response.json()["message"] == "Portfolio deleted"
    assert list_response.json()["count"] == 0


def test_history_returns_auto_saved_analysis_runs(monkeypatch: object) -> None:
    # Verifies successful analyze requests are persisted to history so clients can retrieve recent analysis runs later.
    monkeypatch.setattr(
        "app.api.main.run_risk_pipeline", lambda portfolio: _mock_risk_results()
    )

    first_response = client.post("/analyze", json=_balanced_portfolio_payload())
    second_response = client.post("/analyze", json=_balanced_portfolio_payload())
    history_response = client.get("/history")
    body = history_response.json()

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert history_response.status_code == 200
    assert body["count"] == 2
    assert body["runs"][0]["var_95"] == pytest.approx(0.0182)
    assert body["runs"][0]["tickers"] == _balanced_portfolio_payload()["tickers"]
