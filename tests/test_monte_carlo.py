import numpy as np
import pandas as pd

from app.services.monte_carlo import run_monte_carlo_simulation
from app.services.risk_metrics import compute_es, compute_var


def _sample_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAPL": [0.010, 0.020, -0.010, 0.015, 0.005, 0.011],
            "MSFT": [0.012, 0.018, -0.008, 0.014, 0.006, 0.010],
            "SPY": [0.008, 0.010, -0.006, 0.009, 0.004, 0.007],
            "GLD": [0.004, -0.002, 0.003, 0.001, 0.002, 0.000],
        }
    )


def test_output_shape() -> None:
    # Checks the simulator returns one portfolio result per run because downstream VaR/ES calculations expect a 1D simulation vector.
    returns = _sample_returns()
    simulations = run_monte_carlo_simulation(
        returns=returns,
        weights=[0.25, 0.25, 0.30, 0.20],
        simulations=1000,
        horizon_days=1,
    )
    assert simulations.shape == (1000,)


def test_reproducibility() -> None:
    # Checks a fixed random seed reproduces identical paths because risk analysis should be debuggable and deterministic in tests.
    returns = _sample_returns()
    first_run = run_monte_carlo_simulation(
        returns=returns,
        weights=[0.25, 0.25, 0.30, 0.20],
        simulations=1000,
        horizon_days=1,
        random_seed=123,
    )
    second_run = run_monte_carlo_simulation(
        returns=returns,
        weights=[0.25, 0.25, 0.30, 0.20],
        simulations=1000,
        horizon_days=1,
        random_seed=123,
    )
    assert np.array_equal(first_run, second_run)


def test_es_gte_var() -> None:
    # Checks expected shortfall is at least as large as VaR because the average tail loss should not be smaller than the cutoff loss.
    returns = _sample_returns()
    simulations = run_monte_carlo_simulation(
        returns=returns,
        weights=[0.25, 0.25, 0.30, 0.20],
        simulations=5000,
        horizon_days=1,
        random_seed=42,
    )
    var_95 = compute_var(simulations, confidence_level=0.95)
    es_95 = compute_es(simulations, confidence_level=0.95)
    assert es_95 >= var_95
