from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel
import uvicorn

from app.models.api_models import (
    AnalyzeRequest,
    AnalyzeResponse,
    CorrelationMatrix,
    ErrorResponse,
    SamplePortfoliosResponse,
    SimulationResponse,
)
from app.models.portfolio import PortfolioInput
from app.models.results import RiskResults
from app.services.market_data import fetch_price_data
from app.services.monte_carlo import run_monte_carlo_simulation
from app.services.pipeline import run_risk_pipeline
from app.services.portfolio_math import compute_daily_returns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


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

SAMPLE_PORTFOLIOS_PATH = Path("data/sample_portfolios.json")
try:
    SAMPLE_PORTFOLIOS_DATA = json.loads(SAMPLE_PORTFOLIOS_PATH.read_text())
except FileNotFoundError:
    SAMPLE_PORTFOLIOS_DATA = None


@app.middleware("http")
async def log_requests(request: Request, call_next) -> JSONResponse:
    """Log incoming requests and outgoing responses for basic API observability.

    Args:
        request: The incoming FastAPI request object.
        call_next: The next ASGI handler in the request pipeline.

    Returns:
        The response produced by downstream handlers.
    """

    request_timestamp = datetime.now(timezone.utc).isoformat()
    start_time = perf_counter()
    logger.info(
        "Incoming request | method=%s path=%s timestamp=%s",
        request.method,
        request.url.path,
        request_timestamp,
    )
    response = await call_next(request)
    duration_ms = (perf_counter() - start_time) * 1000
    logger.info(
        "Outgoing response | method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


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
def read_root() -> dict[str, str]:
    """Return a welcome message and point users to the interactive API docs.

    Returns:
        A small JSON payload describing the API and where to find the docs UI.
    """

    return {
        "message": "Welcome to the Portfolio Risk Analytics API.",
        "docs": "Visit /docs for the OpenAPI UI.",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def read_health() -> HealthResponse:
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
def read_sample_portfolios() -> SamplePortfoliosResponse | JSONResponse:
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
def analyze_portfolio(request: AnalyzeRequest) -> AnalyzeResponse | JSONResponse:
    """Run the portfolio risk pipeline and return API-shaped analytics results.

    Args:
        request: API request body containing portfolio inputs and simulation settings.

    Returns:
        An AnalyzeResponse on success, or a JSONResponse containing a structured
        error payload when validation or runtime failures occur.
    """

    try:
        portfolio_input = PortfolioInput(
            tickers=request.tickers,
            weights=request.weights,
            start_date=request.start_date,
            end_date=request.end_date,
            confidence_level=request.confidence_level,
            simulations=request.simulations,
            horizon_days=request.horizon_days,
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

        return AnalyzeResponse(
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
            random_seed=request.random_seed,
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


@app.post(
    "/simulate",
    response_model=SimulationResponse,
    response_model_exclude_none=True,
    status_code=200,
    tags=["Simulation"],
)
def simulate_portfolio(request: AnalyzeRequest) -> SimulationResponse | JSONResponse:
    """Run Monte Carlo simulation and return richer distribution statistics.

    Args:
        request: API request body containing portfolio inputs and simulation settings.

    Returns:
        A SimulationResponse on success, or a JSONResponse containing a structured
        error payload when validation or runtime failures occur.
    """

    try:
        prices = fetch_price_data(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        daily_returns = compute_daily_returns(prices)
        simulated_returns = run_monte_carlo_simulation(
            returns=daily_returns,
            weights=request.weights,
            simulations=request.simulations,
            horizon_days=request.horizon_days,
            random_seed=request.random_seed,
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
            tickers=request.tickers,
            simulation_count=request.simulations,
            horizon_days=request.horizon_days,
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
