from typing import List

from pydantic import BaseModel, model_validator, ValidationError

from app.config import DEFAULTS


class PortfolioInput(BaseModel):
    """Pydantic model representing portfolio input parameters.

    Fields:
        tickers: list[str] - asset tickers, e.g. ["AAPL", "MSFT"]
        weights: list[float] - portfolio weights that must sum to 1.0 (within tol)
        start_date: str - analysis window start date (YYYY-MM-DD)
        end_date: str - analysis window end date (YYYY-MM-DD)
        confidence_level: float - VaR/ES confidence level (0.80 - 0.99)
        simulations: int - number of Monte Carlo simulations (1000 - 100000)
        horizon_days: int - holding horizon in days (1 - 30)

    Validation rules (enforced after model creation):
        - len(tickers) == len(weights)
        - all weights > 0
        - weights sum to 1.0 within 0.001 tolerance
        - confidence_level between 0.80 and 0.99
        - simulations between 1000 and 100000
        - horizon_days between 1 and 30
    """

    tickers: List[str]
    weights: List[float]
    start_date: str = DEFAULTS.DEFAULT_START_DATE
    end_date: str = DEFAULTS.DEFAULT_END_DATE
    confidence_level: float = DEFAULTS.DEFAULT_CONFIDENCE_LEVEL
    simulations: int = DEFAULTS.DEFAULT_SIMULATIONS
    horizon_days: int = DEFAULTS.DEFAULT_HORIZON_DAYS

    @model_validator(mode="after")
    def _validate_consistency(self) -> "PortfolioInput":
        """Run cross-field validation after the model is created.

        Raises:
            ValueError: if any validation rule fails with a descriptive message.

        Returns:
            PortfolioInput: the validated model instance (self) when checks pass.
        """

        # Check lengths
        if len(self.tickers) != len(self.weights):
            raise ValueError(
                f"Length mismatch: {len(self.tickers)} tickers but {len(self.weights)} weights."
            )

        # Check positivity of weights
        if not all((w > 0.0 for w in self.weights)):
            raise ValueError("All weights must be positive numbers greater than zero.")

        # Check sum-to-one within tolerance
        total_weight = float(sum(self.weights))
        tol = 0.001
        if abs(total_weight - 1.0) > tol:
            raise ValueError(
                f"Weights must sum to 1.0 within a tolerance of {tol}. Sum is {total_weight:.6f}."
            )

        # Confidence level bounds
        if not (0.80 <= float(self.confidence_level) <= 0.99):
            raise ValueError(
                f"confidence_level must be between 0.80 and 0.99. Got {self.confidence_level}"
            )

        # Simulations bounds
        if not (1000 <= int(self.simulations) <= 100000):
            raise ValueError(
                f"simulations must be between 1000 and 100000. Got {self.simulations}"
            )

        # Horizon bounds
        if not (1 <= int(self.horizon_days) <= 30):
            raise ValueError(
                f"horizon_days must be between 1 and 30. Got {self.horizon_days}"
            )

        return self


__all__ = ["PortfolioInput"]
