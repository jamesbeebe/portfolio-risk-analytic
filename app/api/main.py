from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi import Request
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
    request: Request, payload: AnalyzeRequest
) -> AnalyzeResponse | JSONResponse:
    """Run the portfolio risk pipeline and return API-shaped analytics results.

    Args:
        request: FastAPI request object used by the rate-limiting layer.
        payload: API request body containing portfolio inputs and simulation settings.

    Returns:
        An AnalyzeResponse on success, or a JSONResponse containing a structured
        error payload when validation or runtime failures occur.
    """

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
            random_seed=payload.random_seed,
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
