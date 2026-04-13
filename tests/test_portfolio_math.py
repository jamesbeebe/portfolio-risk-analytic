import pandas as pd
import pytest

from app.models.portfolio import PortfolioInput
from app.services.portfolio_math import (
    compute_correlation_matrix,
    compute_covariance_matrix,
)


def _sample_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAPL": [0.010, 0.020, -0.010, 0.015, 0.005],
            "MSFT": [0.012, 0.018, -0.008, 0.014, 0.006],
            "SPY": [0.008, 0.010, -0.006, 0.009, 0.004],
            "GLD": [0.004, -0.002, 0.003, 0.001, 0.002],
        }
    )


def test_weights_sum_to_one() -> None:
    # Checks valid portfolio weights pass model validation because the pipeline depends on trusted inputs.
    portfolio = PortfolioInput(
        tickers=["AAPL", "MSFT", "SPY"],
        weights=[0.4, 0.4, 0.2],
    )
    assert portfolio.weights == [0.4, 0.4, 0.2]


def test_weights_do_not_sum_to_one() -> None:
    # Checks invalid total weights are rejected because portfolio math assumes a normalized allocation.
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        PortfolioInput(
            tickers=["AAPL", "MSFT", "SPY"],
            weights=[0.5, 0.5, 0.5],
        )


def test_negative_weight() -> None:
    # Checks negative weights are rejected because this phase only supports long-only portfolios.
    with pytest.raises(ValueError, match="All weights must be positive"):
        PortfolioInput(
            tickers=["AAPL", "MSFT", "SPY"],
            weights=[0.6, -0.2, 0.6],
        )


def test_ticker_count_mismatch() -> None:
    # Checks ticker and weight counts must match because each asset needs exactly one portfolio weight.
    with pytest.raises(ValueError, match="Length mismatch"):
        PortfolioInput(
            tickers=["AAPL", "MSFT", "SPY"],
            weights=[0.5, 0.5],
        )


def test_correlation_matrix_shape() -> None:
    # Checks the correlation output preserves asset dimensionality because downstream reporting expects one row and column per ticker.
    returns = _sample_returns()
    correlation_matrix = compute_correlation_matrix(returns)
    assert correlation_matrix.shape == (4, 4)


def test_covariance_matrix_diagonal_positive() -> None:
    # Checks each asset variance is positive because a realistic return series should not produce negative or zero variance here.
    returns = _sample_returns()
    covariance_matrix = compute_covariance_matrix(returns)
    assert (covariance_matrix.values.diagonal() > 0).all()
