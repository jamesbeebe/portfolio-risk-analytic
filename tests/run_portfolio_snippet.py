from pathlib import Path
import sys

from pydantic import ValidationError

# Allow direct execution via `python tests/run_portfolio_snippet.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.portfolio import PortfolioInput

def main() -> None:
    # 1) Create a valid PortfolioInput and print it
    try:
        valid = PortfolioInput(
            tickers=["AAPL", "MSFT", "SPY"],
            weights=[0.4, 0.4, 0.2],
            # other fields use defaults from app.config
        )
        print("Valid PortfolioInput created:")
        print(valid.model_dump_json(indent=2))
    except ValidationError as e:
        print("ValidationError when creating valid portfolio:")
        print(e.json())

    # 2) Try to create one with weights that don't sum to 1.0 and print the error
    try:
        invalid = PortfolioInput(
            tickers=["AAPL", "MSFT", "SPY"],
            weights=[0.5, 0.5, 0.5],  # sums to 1.5 -> should fail
        )
        # If validation somehow passes, show it (shouldn't)
        print("Unexpectedly created invalid portfolio:")
        print(invalid.model_dump_json(indent=2))
    except ValidationError as e:
        print("\nExpected failure for invalid weights:")
        print(e)

if __name__ == "__main__":
    main()
