from app.config import DEFAULTS
from app.services.market_data import fetch_price_data
from app.services.portfolio_math import (
    compute_annualized_volatility,
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_daily_returns,
    compute_mean_daily_return,
    compute_portfolio_returns,
)


def main() -> None:
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

    print("\n=== Portfolio Math ===")
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


if __name__ == "__main__":
    main()
