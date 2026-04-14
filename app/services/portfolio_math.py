from __future__ import annotations

from math import sqrt

import pandas as pd


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily percentage returns from a price DataFrame.

    Args:
        prices: Price history indexed by date where each column is a ticker symbol.

    Returns:
        A DataFrame of daily percentage returns with the same columns as `prices`
        and the first NaN row removed.
    """

    daily_returns = prices.pct_change()
    return daily_returns.dropna(how="all")


def compute_portfolio_returns(returns: pd.DataFrame, weights: list[float]) -> pd.Series:
    """Compute daily portfolio returns from asset returns and portfolio weights.

    Args:
        returns: Asset return series indexed by date with one column per ticker.
        weights: Portfolio weights aligned to the column order in `returns`.

    Returns:
        A Series of daily portfolio returns indexed by date.

    Raises:
        ValueError: If the number of weights does not match the number of columns.
    """

    if len(weights) != len(returns.columns):
        raise ValueError(
            "The number of weights must match the number of return columns. "
            f"Got {len(weights)} weights for {len(returns.columns)} assets."
        )

    weight_series = pd.Series(weights, index=returns.columns, dtype=float)
    return returns.mul(weight_series, axis=1).sum(axis=1)


def compute_annualized_volatility(portfolio_returns: pd.Series) -> float:
    """Compute annualized volatility from a daily portfolio return series.

    Args:
        portfolio_returns: Daily portfolio returns indexed by date.

    Returns:
        The annualized volatility as a float, using 252 trading days.
    """

    daily_volatility = portfolio_returns.std()
    return float(daily_volatility * sqrt(252))


def compute_mean_daily_return(portfolio_returns: pd.Series) -> float:
    """Compute the arithmetic mean of daily portfolio returns.

    Args:
        portfolio_returns: Daily portfolio returns indexed by date.

    Returns:
        The mean daily return as a float.
    """

    return float(portfolio_returns.mean())


def compute_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the annualized covariance matrix for asset returns.

    Args:
        returns: Daily asset return series indexed by date.

    Returns:
        A covariance matrix annualized using 252 trading days.
    """

    return returns.cov() * 252


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for asset returns.

    Args:
        returns: Daily asset return series indexed by date.

    Returns:
        A DataFrame containing the Pearson correlation matrix.
    """

    return returns.corr()
