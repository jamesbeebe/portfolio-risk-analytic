from __future__ import annotations

import json
from pathlib import Path

from app.models.portfolio import PortfolioInput
from app.models.results import RiskResults
from app.services.market_data import fetch_price_data, validate_tickers
from app.services.monte_carlo import run_monte_carlo_simulation
from app.services.portfolio_math import (
    compute_annualized_volatility,
    compute_correlation_matrix,
    compute_daily_returns,
    compute_mean_daily_return,
    compute_portfolio_returns,
)
from app.services.risk_metrics import compute_es, compute_var


def run_risk_pipeline(portfolio: PortfolioInput) -> RiskResults:
    """Run the full risk analytics pipeline for a validated portfolio input.

    Args:
        portfolio: Input portfolio definition and simulation settings.

    Returns:
        A `RiskResults` object containing summary metrics and simulation outputs.
    """

    validate_tickers(portfolio.tickers)

    prices = fetch_price_data(
        tickers=portfolio.tickers,
        start_date=portfolio.start_date,
        end_date=portfolio.end_date,
    )
    daily_returns = compute_daily_returns(prices)
    portfolio_returns = compute_portfolio_returns(daily_returns, portfolio.weights)
    mean_daily_return = compute_mean_daily_return(portfolio_returns)
    annualized_volatility = compute_annualized_volatility(portfolio_returns)
    correlation_matrix = compute_correlation_matrix(daily_returns)
    simulated_returns = run_monte_carlo_simulation(
        returns=daily_returns,
        weights=portfolio.weights,
        simulations=portfolio.simulations,
        horizon_days=portfolio.horizon_days,
    )
    var_95 = compute_var(simulated_returns, confidence_level=0.95)
    es_95 = compute_es(simulated_returns, confidence_level=0.95)
    var_99 = compute_var(simulated_returns, confidence_level=0.99)
    es_99 = compute_es(simulated_returns, confidence_level=0.99)

    return RiskResults(
        tickers=portfolio.tickers,
        weights=portfolio.weights,
        mean_daily_return=mean_daily_return,
        annualized_volatility=annualized_volatility,
        var_95=var_95,
        es_95=es_95,
        var_99=var_99,
        es_99=es_99,
        correlation_matrix=correlation_matrix.to_dict(),
        simulation_count=portfolio.simulations,
        horizon_days=portfolio.horizon_days,
    )


if __name__ == "__main__":
    sample_file = Path("data/sample_portfolios.json")
    sample_payload = json.loads(sample_file.read_text())
    first_portfolio = sample_payload["portfolios"][0]
    portfolio_input = PortfolioInput(**first_portfolio)
    results = run_risk_pipeline(portfolio_input)

    print("=== Portfolio Risk Report ===")
    print(f"Tickers: {', '.join(results.tickers)}")
    print(f"Annualized Volatility: {results.annualized_volatility:.1%}")
    print(f"Mean Daily Return: {results.mean_daily_return:.2%}")
    print(f"VaR  (95%): {results.var_95:.1%}")
    print(f"ES   (95%): {results.es_95:.1%}")
    print(f"VaR  (99%): {results.var_99:.1%}")
    print(f"ES   (99%): {results.es_99:.1%}")
