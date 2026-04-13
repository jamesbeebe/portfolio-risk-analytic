from __future__ import annotations

from math import sqrt

import pandas as pd

from app.config import DEFAULTS
from app.services.market_data import fetch_price_data


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


def compute_portfolio_returns(
    returns: pd.DataFrame, weights: list[float]
) -> pd.Series:
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


if __name__ == "__main__":
    sample_tickers = ["AAPL", "MSFT", "SPY", "GLD"]
    sample_weights = [0.25, 0.25, 0.25, 0.25]

    prices = fetch_price_data(
        tickers=sample_tickers,
        start_date=DEFAULTS.DEFAULT_START_DATE,
        end_date=DEFAULTS.DEFAULT_END_DATE,
    )
    daily_returns = compute_daily_returns(prices)
    portfolio_returns = compute_portfolio_returns(daily_returns, sample_weights)
    annualized_volatility = compute_annualized_volatility(portfolio_returns)
    mean_daily_return = compute_mean_daily_return(portfolio_returns)
    covariance_matrix = compute_covariance_matrix(daily_returns)
    correlation_matrix = compute_correlation_matrix(daily_returns)

    print("\nPrices:")
    print(prices.head(3))
    print("\nDaily Returns:")
    print(daily_returns.head(3))
    print("\nPortfolio Returns:")
    print(portfolio_returns.head(3))
    print(f"\nMean Daily Return: {mean_daily_return:.4%}")
    print(f"Annualized Volatility: {annualized_volatility:.1%}")
    print("\nCovariance Matrix:")
    print(covariance_matrix)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
