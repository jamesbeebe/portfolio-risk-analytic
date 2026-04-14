import pandas as pd
import pytest

from app.services.market_data import (
    clear_market_data_cache,
    fetch_price_data,
    validate_tickers,
)


def test_validate_tickers_empty_string() -> None:
    # Checks blank ticker symbols are rejected because market data requests must only contain valid non-empty identifiers.
    with pytest.raises(ValueError, match="must not contain empty strings"):
        validate_tickers(["AAPL", ""])


def test_validate_tickers_duplicate() -> None:
    # Checks duplicate tickers are rejected because repeated assets would distort portfolio weights and analytics.
    with pytest.raises(ValueError, match="Duplicate tickers"):
        validate_tickers(["AAPL", "AAPL"])


def test_fetch_price_data_reuses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Checks repeated identical market-data requests reuse the in-process cache so public demos avoid duplicate yfinance downloads.
    clear_market_data_cache()
    download_calls = {"count": 0}

    def fake_download(*args, **kwargs) -> pd.DataFrame:
        download_calls["count"] += 1
        dates = pd.date_range("2021-01-04", periods=4, freq="B")
        columns = pd.MultiIndex.from_product(
            [["Adj Close"], ["AAPL", "MSFT"]],
            names=[None, None],
        )
        return pd.DataFrame(
            [
                [100.0, 200.0],
                [101.0, 201.0],
                [102.0, 202.0],
                [103.0, 203.0],
            ],
            index=dates,
            columns=columns,
        )

    monkeypatch.setattr("app.services.market_data.yf.download", fake_download)

    first_result = fetch_price_data(["AAPL", "MSFT"], "2021-01-01", "2021-02-01")
    second_result = fetch_price_data(["AAPL", "MSFT"], "2021-01-01", "2021-02-01")

    assert download_calls["count"] == 1
    assert first_result.equals(second_result)

    first_result.iloc[0, 0] = -999.0
    assert second_result.iloc[0, 0] == 100.0

    clear_market_data_cache()


def test_fetch_price_data_does_not_cache_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Checks failed downloads are not cached permanently so a later retry still attempts a fresh market-data fetch.
    clear_market_data_cache()
    download_calls = {"count": 0}

    def fake_download(*args, **kwargs) -> pd.DataFrame:
        download_calls["count"] += 1
        return pd.DataFrame()

    monkeypatch.setattr("app.services.market_data.yf.download", fake_download)

    with pytest.raises(ValueError, match="No market data returned"):
        fetch_price_data(["AAPL", "MSFT"], "2021-01-01", "2021-02-01")

    with pytest.raises(ValueError, match="No market data returned"):
        fetch_price_data(["AAPL", "MSFT"], "2021-01-01", "2021-02-01")

    assert download_calls["count"] == 2

    clear_market_data_cache()
