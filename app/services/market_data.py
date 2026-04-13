from __future__ import annotations

import pandas as pd
import yfinance as yf


def validate_tickers(tickers: list[str]) -> None:
    """Validate the list of ticker symbols before requesting market data."""

    if any((not ticker.strip() for ticker in tickers)):
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


def fetch_price_data(
    tickers: list[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch and clean adjusted close prices for the requested tickers."""

    validate_tickers(tickers)

    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

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
                price_data.columns = tickers
        else:
            price_data = raw_data.copy()
            if len(tickers) == 1 and price_data.shape[1] == 1:
                price_data.columns = tickers

    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame(name=tickers[0])

    if price_data.empty:
        ticker_list = ", ".join(tickers)
        raise ValueError(
            f"No adjusted close price data available for tickers: {ticker_list}"
        )

    price_data = price_data.reindex(columns=tickers)

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
            "All tickers were dropped after missing-value checks for: "
            f"{ticker_list}"
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
    price_data = price_data.sort_index()

    return price_data


if __name__ == "__main__":
    demo_tickers = ["AAPL", "MSFT", "SPY", "GLD"]
    demo_start_date = "2021-01-01"
    demo_end_date = "2026-01-01"

    prices = fetch_price_data(
        tickers=demo_tickers,
        start_date=demo_start_date,
        end_date=demo_end_date,
    )
    print(f"Price data shape: {prices.shape}")
    print(prices.head(3))
