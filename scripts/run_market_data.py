from app.services.market_data import fetch_price_data


def main() -> None:
    demo_tickers = ["AAPL", "MSFT", "SPY", "GLD"]
    demo_start_date = "2021-01-01"
    demo_end_date = "2026-01-01"

    prices = fetch_price_data(
        tickers=demo_tickers,
        start_date=demo_start_date,
        end_date=demo_end_date,
    )
    print("\n=== Market Data ===")
    print(f"Price data shape: {prices.shape}")
    print(prices.head(3))


if __name__ == "__main__":
    main()
