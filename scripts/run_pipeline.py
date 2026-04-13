import json
from pathlib import Path

from app.models.portfolio import PortfolioInput
from app.services.pipeline import run_risk_pipeline


def main() -> None:
    sample_file = Path("data/sample_portfolios.json")
    sample_payload = json.loads(sample_file.read_text())
    first_portfolio = sample_payload["portfolios"][0]
    portfolio_input = PortfolioInput(**first_portfolio)
    results = run_risk_pipeline(portfolio_input)

    print("\n=== Portfolio Risk Report ===")
    print(f"Tickers: {', '.join(results.tickers)}")
    print(f"Annualized Volatility: {results.annualized_volatility:.1%}")
    print(f"Mean Daily Return: {results.mean_daily_return:.2%}")
    print(f"VaR  (95%): {results.var_95:.1%}")
    print(f"ES   (95%): {results.es_95:.1%}")
    print(f"VaR  (99%): {results.var_99:.1%}")
    print(f"ES   (99%): {results.es_99:.1%}")


if __name__ == "__main__":
    main()
