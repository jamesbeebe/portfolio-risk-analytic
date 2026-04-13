import pytest

from app.services.market_data import validate_tickers


def test_validate_tickers_empty_string() -> None:
    # Checks blank ticker symbols are rejected because market data requests must only contain valid non-empty identifiers.
    with pytest.raises(ValueError, match="must not contain empty strings"):
        validate_tickers(["AAPL", ""])


def test_validate_tickers_duplicate() -> None:
    # Checks duplicate tickers are rejected because repeated assets would distort portfolio weights and analytics.
    with pytest.raises(ValueError, match="Duplicate tickers"):
        validate_tickers(["AAPL", "AAPL"])
