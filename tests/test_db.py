"""Database-layer tests using an isolated in-memory SQLite database.

We use a separate in-memory database for testing so these tests never touch the
real local SQLite file or any future deployed database. In-memory SQLite uses
the special connection string `sqlite://`, which creates a temporary database
that lives only in RAM for the duration of the test session/connection.

Because each test gets a fresh schema and disposable data, the tests stay
isolated, repeatable, and safe to run in any environment without cleanup risk.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.db import crud
from app.db.database import Base, get_db
from app.models.api_models import AnalyzeRequest, AnalyzeResponse


@pytest.fixture()
def test_db() -> Session:
    """Create a fresh in-memory SQLite session for one test.

    Returns:
        A live SQLAlchemy session backed by an in-memory SQLite database.
    """

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Dropping tables after each test guarantees no rows or schema state can
        # leak into the next test, which keeps the suite deterministic.
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client_with_test_db(test_db: Session) -> TestClient:
    """Provide a FastAPI test client wired to the in-memory database.

    Args:
        test_db: SQLAlchemy session bound to the in-memory SQLite engine.

    Returns:
        A TestClient whose database dependency points at the test session.
    """

    previous_enabled = app.state.limiter.enabled
    app.state.limiter.enabled = False

    def override_get_db():
        """Yield the in-memory test database session to FastAPI routes."""

        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    app.state.limiter.enabled = previous_enabled


def _portfolio_request(name_seed: str = "Base") -> AnalyzeRequest:
    """Build a valid AnalyzeRequest object for persistence tests.

    Args:
        name_seed: Small differentiator for test readability.

    Returns:
        A valid AnalyzeRequest model instance.
    """

    ticker_map = {
        "Base": ["AAPL", "MSFT", "SPY", "GLD"],
        "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL"],
    }
    tickers = ticker_map.get(name_seed, ["AAPL", "MSFT", "SPY", "GLD"])
    return AnalyzeRequest(
        tickers=tickers,
        weights=[0.25, 0.25, 0.30, 0.20],
        start_date="2021-01-01",
        end_date="2026-01-01",
        confidence_level=0.95,
        simulations=10000,
        horizon_days=1,
        random_seed=42,
    )


def _analysis_response() -> AnalyzeResponse:
    """Build a realistic AnalyzeResponse object for history tests.

    Returns:
        A completed AnalyzeResponse model instance.
    """

    return AnalyzeResponse(
        tickers=["AAPL", "MSFT", "SPY", "GLD"],
        weights=[0.25, 0.25, 0.30, 0.20],
        mean_daily_return=0.0007,
        annualized_volatility=0.1243,
        var_95=0.0182,
        es_95=0.0241,
        var_99=0.0274,
        es_99=0.0347,
        correlation={
            "tickers": ["AAPL", "MSFT", "SPY", "GLD"],
            "matrix": [
                [1.0, 0.86, 0.78, 0.11],
                [0.86, 1.0, 0.76, 0.09],
                [0.78, 0.76, 1.0, 0.04],
                [0.11, 0.09, 0.04, 1.0],
            ],
        },
        simulation_count=10000,
        horizon_days=1,
        random_seed=42,
    )


def test_save_and_retrieve_portfolio(test_db: Session) -> None:
    # Verifies a saved portfolio can be written and then read back with matching core fields.
    portfolio = _portfolio_request()
    saved = crud.save_portfolio(
        db=test_db,
        name="Balanced ETF",
        portfolio=portfolio,
        notes="Core allocation",
    )

    retrieved = crud.get_portfolio_by_id(test_db, saved.id)

    assert retrieved is not None
    assert retrieved.name == "Balanced ETF"
    assert retrieved.tickers == ["AAPL", "MSFT", "SPY", "GLD"]
    assert retrieved.weights == [0.25, 0.25, 0.3, 0.2]
    assert retrieved.created_at is not None


def test_get_all_portfolios_returns_newest_first(test_db: Session) -> None:
    # Verifies saved portfolios are returned with the most recently created row first.
    first = crud.save_portfolio(test_db, "First Portfolio", _portfolio_request())
    second = crud.save_portfolio(test_db, "Second Portfolio", _portfolio_request("Tech"))

    portfolios = crud.get_all_portfolios(test_db)

    assert len(portfolios) == 2
    assert portfolios[0].id == second.id
    assert portfolios[1].id == first.id


def test_delete_portfolio(test_db: Session) -> None:
    # Verifies deleting a saved portfolio returns True for a real row and False for a missing one.
    saved = crud.save_portfolio(test_db, "Balanced ETF", _portfolio_request())

    deleted = crud.delete_portfolio(test_db, saved.id)
    retrieved = crud.get_portfolio_by_id(test_db, saved.id)
    missing_delete = crud.delete_portfolio(test_db, 9999)

    assert deleted is True
    assert retrieved is None
    assert missing_delete is False


def test_save_analysis_run(test_db: Session) -> None:
    # Verifies an analysis run can be persisted and then read back from history with matching metrics.
    crud.save_analysis_run(
        db=test_db,
        result=_analysis_response(),
        duration_ms=42,
    )

    history = crud.get_analysis_history(test_db)

    assert len(history) == 1
    assert history[0].var_95 == pytest.approx(0.0182)


def test_analysis_history_limit(test_db: Session) -> None:
    # Verifies the history query respects its limit so callers never fetch unbounded run lists.
    for _ in range(5):
        crud.save_analysis_run(db=test_db, result=_analysis_response())

    history = crud.get_analysis_history(test_db, limit=3)

    assert len(history) == 3


def test_deserialize_portfolio_to_request(test_db: Session) -> None:
    # Verifies a saved ORM portfolio can be converted back into an AnalyzeRequest with Python lists.
    saved = crud.save_portfolio(test_db, "Balanced ETF", _portfolio_request())

    deserialized = crud.deserialize_portfolio_to_request(saved)

    assert deserialized.tickers == ["AAPL", "MSFT", "SPY", "GLD"]
    assert deserialized.weights == [0.25, 0.25, 0.3, 0.2]
    assert isinstance(deserialized.tickers, list)
    assert isinstance(deserialized.weights, list)


def test_save_portfolio_route(client_with_test_db: TestClient) -> None:
    # Verifies the save route persists a portfolio through the real FastAPI layer with the DB dependency overridden.
    response = client_with_test_db.post(
        "/portfolios/save",
        json={
            "name": "Balanced ETF",
            "notes": "Core allocation",
            "portfolio": _portfolio_request().model_dump(),
        },
    )

    assert response.status_code == 200
    assert "id" in response.json()


def test_get_portfolios_route(client_with_test_db: TestClient) -> None:
    # Verifies the list route returns at least one saved portfolio after a route-level save request.
    client_with_test_db.post(
        "/portfolios/save",
        json={
            "name": "Balanced ETF",
            "portfolio": _portfolio_request().model_dump(),
        },
    )

    response = client_with_test_db.get("/portfolios")
    body = response.json()

    assert response.status_code == 200
    assert body["count"] > 0
    assert len(body["portfolios"]) > 0


def test_delete_portfolio_route(client_with_test_db: TestClient) -> None:
    # Verifies the delete route removes a saved portfolio and a later lookup returns 404.
    save_response = client_with_test_db.post(
        "/portfolios/save",
        json={
            "name": "Balanced ETF",
            "portfolio": _portfolio_request().model_dump(),
        },
    )
    portfolio_id = save_response.json()["id"]

    delete_response = client_with_test_db.delete(f"/portfolios/{portfolio_id}")
    get_response = client_with_test_db.get(f"/portfolios/{portfolio_id}")

    assert delete_response.status_code == 200
    assert get_response.status_code == 404


def test_history_route(client_with_test_db: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Verifies running analyze through the API auto-saves a history row that can be retrieved from /history.
    from app.models.results import RiskResults

    monkeypatch.setattr(
        "app.api.main.run_risk_pipeline",
        lambda portfolio: RiskResults(
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
        ),
    )

    analyze_response = client_with_test_db.post(
        "/analyze",
        json=_portfolio_request().model_dump(),
    )
    history_response = client_with_test_db.get("/history")
    body = history_response.json()

    assert analyze_response.status_code == 200
    assert history_response.status_code == 200
    assert body["count"] >= 1
    assert len(body["runs"]) >= 1
