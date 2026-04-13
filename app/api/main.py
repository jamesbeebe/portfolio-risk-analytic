from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import uvicorn

from app.config import (
    ANALYZE_RATE_LIMIT,
    HEALTH_RATE_LIMIT,
    get_allowed_cors_origins,
    RATE_LIMIT_RETRY_AFTER_SECONDS,
    ROOT_RATE_LIMIT,
    SAMPLE_PORTFOLIOS_RATE_LIMIT,
    SIMULATE_RATE_LIMIT,
)
from app.db import crud
from app.db.database import get_db
from app.db.models import AnalysisRun, SavedPortfolio
from app.models.api_models import (
    AnalysisRunResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    CorrelationMatrix,
    ErrorResponse,
    HistoryResponse,
    PortfolioListResponse,
    SamplePortfoliosResponse,
    SavedPortfolioResponse,
    SavePortfolioRequest,
    SimulationResponse,
)
from app.models.portfolio import PortfolioInput
from app.models.results import RiskResults
from app.services.market_data import fetch_price_data
from app.services.monte_carlo import run_monte_carlo_simulation
from app.services.pipeline import run_risk_pipeline
from app.services.portfolio_math import compute_daily_returns
from sqlalchemy.orm import Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address, default_limits=[])


class HealthResponse(BaseModel):
    """Response model for the health-check endpoint.

    Attributes:
        status: Indicates whether the API is healthy.
        version: Current API version string.
        message: Human-readable message describing API status.
    """

    status: str
    version: str
    message: str


app = FastAPI(
    title="Portfolio Risk Analytics API",
    description=(
        "Computes Monte Carlo VaR, Expected Shortfall, and portfolio statistics"
    ),
    version="0.1.0",
)
app.state.limiter = limiter
allowed_cors_origins = get_allowed_cors_origins()

# Explicit CORS origins are safer than a wildcard for a public demo because they
# restrict browser-based API access to known frontend apps while preserving local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

SAMPLE_PORTFOLIOS_PATH = Path("data/sample_portfolios.json")
try:
    SAMPLE_PORTFOLIOS_DATA = json.loads(SAMPLE_PORTFOLIOS_PATH.read_text())
except FileNotFoundError:
    SAMPLE_PORTFOLIOS_DATA = None

logger.info("Configured CORS origins: %s", allowed_cors_origins)


# FastAPI's Depends() pattern lets the framework inject shared resources into
# route handlers. In plain English, FastAPI will automatically create a database
# session for each request and close it when the request is done.
def serialize_saved_portfolio(portfolio: SavedPortfolio) -> SavedPortfolioResponse:
    """Convert a SavedPortfolio ORM row into the API response shape.

    Args:
        portfolio: Saved portfolio ORM object loaded from the database.

    Returns:
        A SavedPortfolioResponse with JSON fields deserialized into Python lists.
    """

    return SavedPortfolioResponse(
        id=portfolio.id,
        name=portfolio.name,
        tickers=json.loads(portfolio.tickers),
        weights=json.loads(portfolio.weights),
        created_at=portfolio.created_at.isoformat(),
        notes=portfolio.notes,
    )


def serialize_analysis_run(run: AnalysisRun) -> AnalysisRunResponse:
    """Convert an AnalysisRun ORM row into the API response shape.

    Args:
        run: Analysis history ORM object loaded from the database.

    Returns:
        An AnalysisRunResponse with JSON fields deserialized into Python lists.
    """

    return AnalysisRunResponse(
        id=run.id,
        tickers=json.loads(run.tickers),
        weights=json.loads(run.weights),
        mean_daily_return=run.mean_daily_return,
        annualized_volatility=run.annualized_volatility,
        var_95=run.var_95,
        es_95=run.es_95,
        var_99=run.var_99,
        es_99=run.es_99,
        simulation_count=run.simulation_count,
        ran_at=run.ran_at.isoformat(),
        duration_ms=run.duration_ms,
        portfolio_id=run.portfolio_id,
    )


@app.middleware("http")
async def log_requests(request: Request, call_next) -> JSONResponse:
    """Log incoming requests and outgoing responses for API observability.

    Args:
        request: The incoming FastAPI request object.
        call_next: The next ASGI handler in the request pipeline.

    Returns:
        The response produced by downstream handlers.
    """

    request_timestamp = datetime.now(timezone.utc).isoformat()
    start_time = perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    is_expensive_endpoint = request.url.path in {"/analyze", "/simulate"}

    # Log request metadata only. We intentionally do not log request bodies so
    # local development logs stay readable and do not expose user-submitted input.
    logger.info(
        "Incoming request | method=%s path=%s client_ip=%s expensive=%s timestamp=%s",
        request.method,
        request.url.path,
        client_ip,
        is_expensive_endpoint,
        request_timestamp,
    )
    response = await call_next(request)
    duration_ms = (perf_counter() - start_time) * 1000
    logger.info(
        "Outgoing response | method=%s path=%s client_ip=%s expensive=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        client_ip,
        is_expensive_endpoint,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(RateLimitExceeded)
async def handle_rate_limit_exceeded(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return a professional 429 response when an endpoint limit is exceeded.

    Args:
        request: The incoming FastAPI request object.
        exc: The slowapi rate-limit exception raised for the request.

    Returns:
        A JSONResponse containing a readable rate-limit error payload.
    """

    logger.warning(
        "Rate limit exceeded | method=%s path=%s detail=%s",
        request.method,
        request.url.path,
        exc.detail,
    )
    error_response = ErrorResponse(
        error="rate_limit_exceeded",
        detail=(
            "Rate limit exceeded for this endpoint. Please wait about "
            f"{RATE_LIMIT_RETRY_AFTER_SECONDS} seconds before trying again."
        ),
        field=None,
    )
    return JSONResponse(
        status_code=429,
        content=error_response.model_dump(),
        headers={"Retry-After": str(RATE_LIMIT_RETRY_AFTER_SECONDS)},
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a standardized error payload for FastAPI request validation failures.

    Args:
        request: The incoming FastAPI request object.
        exc: The validation error raised before the route handler executes.

    Returns:
        A JSONResponse containing a normalized ErrorResponse payload.
    """

    # This handler is needed because FastAPI validates the request body before
    # entering the route function, so route-level try/except blocks never see
    # these schema and parsing failures.
    first_error = exc.errors()[0] if exc.errors() else {}
    error_location = first_error.get("loc", [])
    field = ".".join(str(item) for item in error_location[1:]) or None
    detail = first_error.get("msg", "Request validation failed.")

    error_response = ErrorResponse(
        error="validation_error",
        detail=detail,
        field=field,
    )
    return JSONResponse(status_code=422, content=error_response.model_dump())


@app.exception_handler(Exception)
async def handle_unexpected_exception(
    request: Request, exc: Exception
) -> JSONResponse:
    """Log and normalize unexpected server exceptions into a 500 response.

    Args:
        request: The incoming FastAPI request object.
        exc: The uncaught exception raised during request processing.

    Returns:
        A JSONResponse containing a generic server-error payload.
    """

    # Logging is better than print in a server application because it provides
    # levels, timestamps, and centralized formatting that work consistently in
    # development, tests, and deployed environments.
    logger.exception(
        "Unhandled exception | method=%s path=%s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    error_response = ErrorResponse(
        error="unexpected_error",
        detail="An unexpected server error occurred.",
    )
    return JSONResponse(status_code=500, content=error_response.model_dump())


@app.get("/", tags=["Root"])
@limiter.limit(ROOT_RATE_LIMIT)
def read_root(request: Request) -> dict[str, str]:
    """Return a welcome message and point users to the interactive API docs.

    Returns:
        A small JSON payload describing the API and where to find the docs UI.
    """

    return {
        "message": "Welcome to the Portfolio Risk Analytics API.",
        "docs": "Visit /docs for the OpenAPI UI.",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit(HEALTH_RATE_LIMIT)
def read_health(request: Request) -> HealthResponse:
    """Return a simple health-check payload for service monitoring.

    Returns:
        A HealthResponse object confirming the API is running.
    """

    return HealthResponse(
        status="ok",
        version="0.1.0",
        message="Portfolio Risk API is running",
    )


@app.get(
    "/sample-portfolios",
    response_model=SamplePortfoliosResponse,
    response_model_exclude_none=True,
    status_code=200,
    tags=["Samples"],
)
@limiter.limit(SAMPLE_PORTFOLIOS_RATE_LIMIT)
def read_sample_portfolios(request: Request) -> SamplePortfoliosResponse | JSONResponse:
    """Return the bundled sample portfolio definitions used for demos.

    Returns:
        A SamplePortfoliosResponse on success, or a JSONResponse describing why
        the sample data is unavailable.
    """

    if SAMPLE_PORTFOLIOS_DATA is None:
        error_response = ErrorResponse(
            error="data_unavailable",
            detail="The sample portfolio data file is unavailable.",
        )
        return JSONResponse(status_code=503, content=error_response.model_dump())

    portfolios = [
        AnalyzeRequest(**portfolio_data)
        for portfolio_data in SAMPLE_PORTFOLIOS_DATA["portfolios"]
    ]
    return SamplePortfoliosResponse(portfolios=portfolios, count=len(portfolios))


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    response_model_exclude_none=True,
    status_code=200,
    tags=["Analysis"],
)
@limiter.limit(ANALYZE_RATE_LIMIT)
def analyze_portfolio(
    request: Request,
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
) -> AnalyzeResponse | JSONResponse:
    """Run the portfolio risk pipeline and return API-shaped analytics results.

    Args:
        request: FastAPI request object used by the rate-limiting layer.
        payload: API request body containing portfolio inputs and simulation settings.

    Returns:
        An AnalyzeResponse on success, or a JSONResponse containing a structured
        error payload when validation or runtime failures occur.
    """

    analysis_start = perf_counter()
    try:
        portfolio_input = PortfolioInput(
            tickers=payload.tickers,
            weights=payload.weights,
            start_date=payload.start_date,
            end_date=payload.end_date,
            confidence_level=payload.confidence_level,
            simulations=payload.simulations,
            horizon_days=payload.horizon_days,
        )

        risk_results: RiskResults = run_risk_pipeline(portfolio_input)

        correlation_tickers = list(risk_results.correlation_matrix.keys())
        correlation_matrix = [
            [
                float(risk_results.correlation_matrix[row_ticker][column_ticker])
                for column_ticker in correlation_tickers
            ]
            for row_ticker in correlation_tickers
        ]

        response_model = AnalyzeResponse(
            tickers=risk_results.tickers,
            weights=risk_results.weights,
            mean_daily_return=risk_results.mean_daily_return,
            annualized_volatility=risk_results.annualized_volatility,
            var_95=risk_results.var_95,
            es_95=risk_results.es_95,
            var_99=risk_results.var_99,
            es_99=risk_results.es_99,
            correlation=CorrelationMatrix(
                tickers=correlation_tickers,
                matrix=correlation_matrix,
            ),
            simulation_count=risk_results.simulation_count,
            horizon_days=risk_results.horizon_days,
            random_seed=payload.random_seed,
        )
        duration_ms = int((perf_counter() - analysis_start) * 1000)

        try:
            # A database write failure should never prevent the user from getting
            # their analysis result, so persistence is handled separately here.
            crud.save_analysis_run(
                db=db,
                result=response_model,
                duration_ms=duration_ms,
            )
        except Exception:
            logger.exception("Failed to save analysis history for /analyze request.")

        return response_model
    except ValueError as exc:
        error_response = ErrorResponse(
            error="validation_error",
            detail=str(exc),
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())
    except KeyError as exc:
        error_response = ErrorResponse(
            error="data_error",
            detail=f"Missing expected data field: {exc}",
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())
    except Exception as exc:
        error_response = ErrorResponse(
            error="internal_error",
            detail=f"Unexpected server error: {exc}",
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())


@app.post(
    "/portfolios/save",
    response_model=SavedPortfolioResponse,
    response_model_exclude_none=True,
    status_code=200,
    tags=["Persistence"],
)
def save_portfolio_route(
    payload: SavePortfolioRequest,
    db: Session = Depends(get_db),
) -> SavedPortfolioResponse | JSONResponse:
    """Persist a named portfolio preset for later reuse.

    Args:
        payload: Request body containing the portfolio, display name, and notes.
        db: Active SQLAlchemy database session supplied by FastAPI.

    Returns:
        A SavedPortfolioResponse on success, or a JSONResponse on failure.
    """

    try:
        saved_portfolio = crud.save_portfolio(
            db=db,
            name=payload.name,
            portfolio=payload.portfolio,
            notes=payload.notes,
        )
        return serialize_saved_portfolio(saved_portfolio)
    except Exception:
        logger.exception("Failed to save portfolio.")
        error_response = ErrorResponse(
            error="internal_error",
            detail="Failed to save the portfolio.",
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())


@app.get(
    "/portfolios",
    response_model=PortfolioListResponse,
    response_model_exclude_none=True,
    status_code=200,
    tags=["Persistence"],
)
def list_portfolios(
    db: Session = Depends(get_db),
) -> PortfolioListResponse | JSONResponse:
    """Return all saved portfolio presets.

    Args:
        db: Active SQLAlchemy database session supplied by FastAPI.

    Returns:
        A PortfolioListResponse on success, or a JSONResponse on failure.
    """

    try:
        portfolios = crud.get_all_portfolios(db)
        serialized = [serialize_saved_portfolio(portfolio) for portfolio in portfolios]
        return PortfolioListResponse(portfolios=serialized, count=len(serialized))
    except Exception:
        logger.exception("Failed to list portfolios.")
        error_response = ErrorResponse(
            error="internal_error",
            detail="Failed to load saved portfolios.",
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())


@app.get(
    "/portfolios/{portfolio_id}",
    response_model=SavedPortfolioResponse,
    response_model_exclude_none=True,
    status_code=200,
    tags=["Persistence"],
)
def get_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
) -> SavedPortfolioResponse | JSONResponse:
    """Return one saved portfolio by database ID.

    Args:
        portfolio_id: Database ID of the requested portfolio.
        db: Active SQLAlchemy database session supplied by FastAPI.

    Returns:
        A SavedPortfolioResponse when found, or a JSONResponse error otherwise.
    """

    try:
        portfolio = crud.get_portfolio_by_id(db, portfolio_id)
        if portfolio is None:
            error_response = ErrorResponse(
                error="not_found",
                detail=f"Portfolio with id {portfolio_id} not found",
            )
            return JSONResponse(status_code=404, content=error_response.model_dump())
        return serialize_saved_portfolio(portfolio)
    except Exception:
        logger.exception("Failed to load portfolio id=%s.", portfolio_id)
        error_response = ErrorResponse(
            error="internal_error",
            detail="Failed to load the requested portfolio.",
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())


@app.delete(
    "/portfolios/{portfolio_id}",
    response_model=None,
    response_model_exclude_none=True,
    status_code=200,
    tags=["Persistence"],
)
def delete_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
) -> dict[str, str] | JSONResponse:
    """Delete one saved portfolio by database ID.

    Args:
        portfolio_id: Database ID of the portfolio to delete.
        db: Active SQLAlchemy database session supplied by FastAPI.

    Returns:
        A success message when deleted, or a JSONResponse error otherwise.
    """

    try:
        deleted = crud.delete_portfolio(db, portfolio_id)
        if not deleted:
            error_response = ErrorResponse(
                error="not_found",
                detail=f"Portfolio with id {portfolio_id} not found",
            )
            return JSONResponse(status_code=404, content=error_response.model_dump())
        return {"message": "Portfolio deleted"}
    except Exception:
        logger.exception("Failed to delete portfolio id=%s.", portfolio_id)
        error_response = ErrorResponse(
            error="internal_error",
            detail="Failed to delete the requested portfolio.",
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())


@app.get(
    "/history",
    response_model=HistoryResponse,
    response_model_exclude_none=True,
    status_code=200,
    tags=["Persistence"],
)
def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> HistoryResponse | JSONResponse:
    """Return recent analysis history rows from the database.

    Args:
        limit: Maximum number of history rows to return.
        db: Active SQLAlchemy database session supplied by FastAPI.

    Returns:
        A HistoryResponse on success, or a JSONResponse on failure.
    """

    try:
        runs = crud.get_analysis_history(db=db, limit=limit)
        serialized = [serialize_analysis_run(run) for run in runs]
        return HistoryResponse(runs=serialized, count=len(serialized))
    except Exception:
        logger.exception("Failed to load analysis history.")
        error_response = ErrorResponse(
            error="internal_error",
            detail="Failed to load analysis history.",
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())


@app.post(
    "/simulate",
    response_model=SimulationResponse,
    response_model_exclude_none=True,
    status_code=200,
    tags=["Simulation"],
)
@limiter.limit(SIMULATE_RATE_LIMIT)
def simulate_portfolio(
    request: Request, payload: AnalyzeRequest
) -> SimulationResponse | JSONResponse:
    """Run Monte Carlo simulation and return richer distribution statistics.

    Args:
        request: FastAPI request object used by the rate-limiting layer.
        payload: API request body containing portfolio inputs and simulation settings.

    Returns:
        A SimulationResponse on success, or a JSONResponse containing a structured
        error payload when validation or runtime failures occur.
    """

    try:
        prices = fetch_price_data(
            tickers=payload.tickers,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
        daily_returns = compute_daily_returns(prices)
        simulated_returns = run_monte_carlo_simulation(
            returns=daily_returns,
            weights=payload.weights,
            simulations=payload.simulations,
            horizon_days=payload.horizon_days,
            random_seed=payload.random_seed,
        )

        percentile_levels = {
            "p1": 1,
            "p5": 5,
            "p10": 10,
            "p25": 25,
            "p50": 50,
            "p75": 75,
            "p90": 90,
            "p95": 95,
            "p99": 99,
        }
        percentiles = {
            label: float(np.percentile(simulated_returns, level))
            for label, level in percentile_levels.items()
        }

        return SimulationResponse(
            tickers=payload.tickers,
            simulation_count=payload.simulations,
            horizon_days=payload.horizon_days,
            percentiles=percentiles,
            mean_return=float(np.mean(simulated_returns)),
            std_dev=float(np.std(simulated_returns)),
            worst_case=float(np.min(simulated_returns)),
            best_case=float(np.max(simulated_returns)),
        )
    except ValueError as exc:
        error_response = ErrorResponse(
            error="validation_error",
            detail=str(exc),
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())
    except KeyError as exc:
        error_response = ErrorResponse(
            error="data_error",
            detail=f"Missing expected data field: {exc}",
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())
    except Exception as exc:
        error_response = ErrorResponse(
            error="internal_error",
            detail=f"Unexpected server error: {exc}",
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
