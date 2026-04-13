from typing import Dict, List

from pydantic import BaseModel


class RiskResults(BaseModel):
    """Model for holding computed risk results for a portfolio.

    Fields:
        tickers: list[str] - the asset tickers in the portfolio
        weights: list[float] - the portfolio weights
        mean_daily_return: float - average daily return (decimal, e.g. 0.0012)
        annualized_volatility: float - annualized volatility (decimal)
        var_95: float - 95% Value at Risk (positive number representing loss %)
        es_95: float - 95% Expected Shortfall (positive number representing loss %)
        var_99: float - 99% Value at Risk
        es_99: float - 99% Expected Shortfall
        correlation_matrix: dict - nested dict mapping ticker -> ticker -> correlation float
        simulation_count: int - number of simulations used
        horizon_days: int - holding horizon in days
    """

    tickers: List[str]
    weights: List[float]
    mean_daily_return: float
    annualized_volatility: float
    var_95: float
    es_95: float
    var_99: float
    es_99: float
    correlation_matrix: Dict[str, Dict[str, float]]
    simulation_count: int
    horizon_days: int


__all__ = ["RiskResults"]
