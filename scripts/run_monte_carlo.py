from app.config import DEFAULTS
from app.services.market_data import fetch_price_data
from app.services.monte_carlo import run_monte_carlo_simulation
from app.services.portfolio_math import compute_daily_returns


def main() -> None:
    sample_tickers = ["AAPL", "MSFT", "SPY", "GLD"]
    sample_weights = [0.25, 0.25, 0.25, 0.25]

    prices = fetch_price_data(
        tickers=sample_tickers,
        start_date=DEFAULTS.DEFAULT_START_DATE,
        end_date=DEFAULTS.DEFAULT_END_DATE,
    )
    daily_returns = compute_daily_returns(prices)
    simulation_results = run_monte_carlo_simulation(
        returns=daily_returns,
        weights=sample_weights,
        simulations=DEFAULTS.DEFAULT_SIMULATIONS,
        horizon_days=DEFAULTS.DEFAULT_HORIZON_DAYS,
    )
    print("\n=== Monte Carlo ===")
    print(f"Simulation Results Shape: {simulation_results.shape}")
    print(f"Minimum Return: {simulation_results.min():.4%}")
    print(f"Maximum Return: {simulation_results.max():.4%}")
    print(f"Mean Return: {simulation_results.mean():.4%}")
    print(f"Standard Deviation: {simulation_results.std():.4%}")


if __name__ == "__main__":
    main()
