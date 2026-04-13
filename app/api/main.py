from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from app.models.api_models import (
    AnalyzeRequest,
    AnalyzeResponse,
    CorrelationMatrix,
    ErrorResponse,
)
from app.models.portfolio import PortfolioInput
from app.models.results import RiskResults
from app.services.pipeline import run_risk_pipeline


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


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
