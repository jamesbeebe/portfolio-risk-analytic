"""Configuration defaults for the portfolio risk platform.

This module provides a frozen dataclass `Config` and a module-level `DEFAULTS`
instance holding default values used across the application.
"""
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
