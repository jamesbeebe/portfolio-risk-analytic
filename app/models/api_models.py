from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.config import DEFAULTS


class AnalyzeRequest(BaseModel):
    """Request model for portfolio risk analysis submitted to the API.

    Attributes:
        tickers: Ordered list of ticker symbols to analyze.
        weights: Portfolio weights aligned to the ticker order.
        start_date: Analysis window start date in YYYY-MM-DD format.
        end_date: Analysis window end date in YYYY-MM-DD format.
        confidence_level: Confidence level used for risk reporting.
        simulations: Number of Monte Carlo simulation paths to generate.
        horizon_days: Number of trading days in the simulated holding period.
        random_seed: Seed used to make simulations reproducible.
    """

    tickers: list[str]
    weights: list[float]
    start_date: str = DEFAULTS.DEFAULT_START_DATE
    end_date: str = DEFAULTS.DEFAULT_END_DATE
    confidence_level: float = Field(
        default=DEFAULTS.DEFAULT_CONFIDENCE_LEVEL,
        ge=0.80,
        le=0.99,
    )
    simulations: int = Field(default=DEFAULTS.DEFAULT_SIMULATIONS, ge=1000, le=100000)
    horizon_days: int = Field(default=DEFAULTS.DEFAULT_HORIZON_DAYS, ge=1, le=30)
    random_seed: int = 42

    @model_validator(mode="after")
    def validate_portfolio_inputs(self) -> "AnalyzeRequest":
        """Validate cross-field portfolio constraints for the API request.

        Returns:
            The validated AnalyzeRequest instance.

        Raises:
            ValueError: If ticker and weight lengths differ, if any weight is not
                positive, or if the weights do not sum to 1.0 within tolerance.
        """

        if len(self.tickers) != len(self.weights):
            raise ValueError(
                "The number of tickers must match the number of weights."
            )

        if not all((weight > 0.0 for weight in self.weights)):
            raise ValueError("All portfolio weights must be positive numbers.")

        total_weight = float(sum(self.weights))
        tolerance = 0.001
        if abs(total_weight - 1.0) > tolerance:
            raise ValueError(
                "Portfolio weights must sum to 1.0 within a tolerance of 0.001."
            )

        return self


class CorrelationMatrix(BaseModel):
    """API-friendly representation of a correlation matrix.

    Attributes:
        tickers: Ordered ticker labels used for both matrix rows and columns.
        matrix: Two-dimensional correlation values aligned to the ticker order.
    """

    tickers: list[str]
    matrix: list[list[float]]


class AnalyzeResponse(BaseModel):
    """Response model returned by the portfolio analysis endpoint.

    Attributes:
        tickers: Ordered list of analyzed tickers.
        weights: Portfolio weights aligned to the ticker order.
        mean_daily_return: Arithmetic mean of historical daily portfolio returns.
        annualized_volatility: Annualized volatility of historical portfolio returns.
        var_95: 95% Value at Risk as a positive loss magnitude.
        es_95: 95% Expected Shortfall as a positive loss magnitude.
        var_99: 99% Value at Risk as a positive loss magnitude.
        es_99: 99% Expected Shortfall as a positive loss magnitude.
        correlation: Correlation matrix in API-friendly list form.
        simulation_count: Number of Monte Carlo simulation paths used.
        horizon_days: Holding period used in the simulation.
        random_seed: Seed used to generate reproducible simulation results.
    """

    tickers: list[str]
    weights: list[float]
    mean_daily_return: float
    annualized_volatility: float
    var_95: float
    es_95: float
    var_99: float
    es_99: float
    correlation: CorrelationMatrix
    simulation_count: int
    horizon_days: int
    random_seed: int


class SamplePortfoliosResponse(BaseModel):
    """Response model for returning bundled sample portfolio definitions.

    Attributes:
        portfolios: List of sample portfolios shaped like analysis requests.
        count: Number of sample portfolios returned.
    """

    portfolios: list[AnalyzeRequest]
    count: int


class SimulationResponse(BaseModel):
    """Response model for detailed Monte Carlo simulation statistics.

    Attributes:
        tickers: Ordered list of analyzed tickers.
        simulation_count: Number of Monte Carlo simulation paths used.
        horizon_days: Holding period used in the simulation.
        percentiles: Portfolio return values at selected percentile levels.
        mean_return: Arithmetic mean of simulated portfolio returns.
        std_dev: Standard deviation of simulated portfolio returns.
        worst_case: Minimum simulated portfolio return.
        best_case: Maximum simulated portfolio return.
    """

    tickers: list[str]
    simulation_count: int
    horizon_days: int
    percentiles: dict[str, float]
    mean_return: float
    std_dev: float
    worst_case: float
    best_case: float


class ErrorResponse(BaseModel):
    """Standard error payload returned by the API when a request fails.

    Attributes:
        error: Short machine-readable error identifier.
        detail: Human-readable description of what went wrong.
        field: Optional field name associated with the error.
    """

    error: str
    detail: str
    field: str | None = None
