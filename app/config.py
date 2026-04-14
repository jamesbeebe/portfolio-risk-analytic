"""Configuration defaults for the portfolio risk platform.

This module provides a frozen dataclass `Config` and a module-level `DEFAULTS`
instance holding default values used across the application.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Holds default configuration values for data windows and simulations.

    Attributes:
        DEFAULT_START_DATE: str - default start date (YYYY-MM-DD)
        DEFAULT_END_DATE: str - default end date (YYYY-MM-DD)
        DEFAULT_CONFIDENCE_LEVEL: float - default VaR/CVaR confidence level
        DEFAULT_SIMULATIONS: int - default number of Monte Carlo simulations
        DEFAULT_HORIZON_DAYS: int - default holding horizon in days
    """

    DEFAULT_START_DATE: str = "2021-01-01"
    DEFAULT_END_DATE: str = "2026-01-01"
    DEFAULT_CONFIDENCE_LEVEL: float = 0.95
    DEFAULT_SIMULATIONS: int = 10000
    DEFAULT_HORIZON_DAYS: int = 1


# Module-level default instance for convenience
DEFAULTS = Config()

# Public demo hardening settings for backend request limiting.
HEALTH_RATE_LIMIT = "60/minute"
ROOT_RATE_LIMIT = "30/minute"
SAMPLE_PORTFOLIOS_RATE_LIMIT = "20/minute"
ANALYZE_RATE_LIMIT = "10/minute"
SIMULATE_RATE_LIMIT = "5/minute"
RATE_LIMIT_RETRY_AFTER_SECONDS = 60

# In-process caching settings for public-demo stability.
MARKET_DATA_CACHE_SIZE = 32

# CORS settings for local development and future deployed frontends.
DEFAULT_CORS_ORIGINS = (
    "http://localhost:8501",
    "http://127.0.0.1:8501",
)
EXTRA_CORS_ORIGINS_ENV_VAR = "EXTRA_CORS_ORIGINS"


def get_allowed_cors_origins() -> list[str]:
    """Build the allowed CORS origin list from defaults plus environment config.

    Returns:
        A de-duplicated list of allowed origins for FastAPI CORS middleware.

    Notes:
        Wildcard CORS (`*`) is convenient for development but is not ideal for a
        public release because it allows any site to call the API from a browser.
        For this project, local development origins are allowed by default and
        additional deployed frontend origins can be appended through an env var.
        Example:
        EXTRA_CORS_ORIGINS=https://my-app.streamlit.app,https://portfolio-risk-api.onrender.com
    """

    extra_origins_raw = os.getenv(EXTRA_CORS_ORIGINS_ENV_VAR, "")
    extra_origins = [
        origin.strip()
        for origin in extra_origins_raw.split(",")
        if origin.strip()
    ]

    allowed_origins: list[str] = []
    for origin in [*DEFAULT_CORS_ORIGINS, *extra_origins]:
        if origin not in allowed_origins:
            allowed_origins.append(origin)

    return allowed_origins
