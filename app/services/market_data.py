from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yfinance as yf

from app.config import MARKET_DATA_CACHE_SIZE


def validate_tickers(tickers: list[str]) -> None:
    """Validate the list of ticker symbols before requesting market data."""

    if any(not ticker.strip() for ticker in tickers):
        raise ValueError("Tickers must not contain empty strings.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for ticker in tickers:
        if ticker in seen and ticker not in duplicates:
            duplicates.append(ticker)
        seen.add(ticker)

    if duplicates:
        duplicate_list = ", ".join(duplicates)
        raise ValueError(f"Duplicate tickers are not allowed: {duplicate_list}")


def _prepare_price_data(
    raw_data: pd.DataFrame, tickers: tuple[str, ...]
) -> pd.DataFrame:
    """Clean raw yfinance output into a date-indexed adjusted-close price table.

    Args:
        raw_data: Raw DataFrame returned by `yfinance.download`.
        tickers: Ordered ticker symbols used for the request.

    Returns:
        A cleaned DataFrame with dates as rows and ticker symbols as columns.

    Raises:
        ValueError: If no usable adjusted close data remains after cleaning.
    """

    if raw_data.empty:
        ticker_list = ", ".join(tickers)
        raise ValueError(
            f"No market data returned by yfinance for tickers: {ticker_list}"
        )

    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Adj Close" not in raw_data.columns.get_level_values(0):
            ticker_list = ", ".join(tickers)
            raise ValueError(
                f"Adjusted close prices were not returned for tickers: {ticker_list}"
            )
        price_data = raw_data["Adj Close"].copy()
    else:
        if "Adj Close" in raw_data.columns:
            price_data = raw_data[["Adj Close"]].copy()
            if len(tickers) == 1:
                price_data.columns = list(tickers)
        else:
            price_data = raw_data.copy()
            if len(tickers) == 1 and price_data.shape[1] == 1:
                price_data.columns = list(tickers)

    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame(name=tickers[0])

    if price_data.empty:
        ticker_list = ", ".join(tickers)
        raise ValueError(
            f"No adjusted close price data available for tickers: {ticker_list}"
        )

    price_data = price_data.reindex(columns=list(tickers))

    missing_ratio = price_data.isna().mean()
    dropped_tickers = missing_ratio[missing_ratio > 0.05].index.tolist()
    if dropped_tickers:
        print(
            "Warning: Dropping tickers with more than 5% missing values: "
            + ", ".join(dropped_tickers)
        )
        price_data = price_data.drop(columns=dropped_tickers)

    if price_data.empty:
        ticker_list = ", ".join(tickers)
        raise ValueError(
            f"All tickers were dropped after missing-value checks for: {ticker_list}"
        )

    price_data = price_data.ffill().bfill()
    price_data = price_data.dropna(how="any")

    if price_data.empty:
        remaining = ", ".join(price_data.columns.tolist()) or ", ".join(tickers)
        raise ValueError(
            "No overlapping clean price history remained after alignment for tickers: "
            f"{remaining}"
        )

    price_data.index = pd.to_datetime(price_data.index)
    return price_data.sort_index()


@lru_cache(maxsize=MARKET_DATA_CACHE_SIZE)
def _fetch_price_data_cached(
    tickers: tuple[str, ...], start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch and cache cleaned historical prices for a specific request key.

    Args:
        tickers: Ordered ticker symbols to download.
        start_date: Inclusive historical start date.
        end_date: Exclusive historical end date used by yfinance.

    Returns:
        A cleaned adjusted-close price DataFrame suitable for analytics.

    Notes:
        This is an in-memory per-process cache. It is cleared on server restart and
        is not shared across workers or machines. Exceptions are not cached, so a
        transient download failure can succeed on a later retry.
    """

    raw_data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )
    return _prepare_price_data(raw_data=raw_data, tickers=tickers)


def clear_market_data_cache() -> None:
    """Clear the in-process market-data cache.

    Returns:
        None. This is primarily useful for tests or manual cache resets.
    """

    _fetch_price_data_cached.cache_clear()


def fetch_price_data(
    tickers: list[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch and clean adjusted close prices for the requested tickers.

    Args:
        tickers: Ordered list of ticker symbols to fetch.
        start_date: Inclusive historical start date in YYYY-MM-DD format.
        end_date: Exclusive historical end date in YYYY-MM-DD format.

    Returns:
        A cleaned price DataFrame with dates as rows and ticker symbols as columns.

    Notes:
        Successful fetches are cached in-process by ticker/date combination to
        reduce repeated yfinance work during public demos. A copy is returned on
        each call so downstream code cannot mutate the cached DataFrame in place.
    """

    validate_tickers(tickers)
    cached_prices = _fetch_price_data_cached(tuple(tickers), start_date, end_date)
    return cached_prices.copy(deep=True)
