from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


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


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
