from __future__ import annotations

"""
Plain-English notes:

- A multivariate normal distribution is a way to simulate several assets together
  so each asset has its own average return and volatility, while also preserving
  how the assets tend to move with or against each other.
- We use a random seed so the simulation is reproducible. With the same inputs
  and the same seed, the simulation produces the same random draws each time.
- The output array represents one simulated portfolio return per simulation. Each
  value is the modeled portfolio return over the requested holding horizon.
"""

import numpy as np
import pandas as pd


def run_monte_carlo_simulation(
    returns: pd.DataFrame,
    weights: list[float],
    simulations: int,
    horizon_days: int,
    random_seed: int = 42,
) -> np.ndarray:
    """Run a Monte Carlo simulation of portfolio returns.

    Args:
        returns: Historical daily asset returns with dates as rows and tickers as
            columns.
        weights: Portfolio weights aligned to the column order in `returns`.
        simulations: Number of Monte Carlo scenarios to generate.
        horizon_days: Number of trading days to simulate for each scenario.
        random_seed: Seed used to make the random simulation reproducible.

    Returns:
        A one-dimensional numpy array of length `simulations` containing one
        simulated portfolio return per scenario.

    Raises:
        ValueError: If the weights do not match the number of assets, or if the
            simulation settings are invalid.
    """

    if len(weights) != len(returns.columns):
        raise ValueError(
            "The number of weights must match the number of return columns. "
            f"Got {len(weights)} weights for {len(returns.columns)} assets."
        )

    if simulations <= 0:
        raise ValueError("simulations must be a positive integer.")

    if horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer.")

    weight_array = np.asarray(weights, dtype=float)

    # 1. Compute the mean return vector (one value per asset) from historical returns.
    mean_vector = returns.mean().to_numpy(dtype=float)

    # 2. Compute the covariance matrix from historical returns.
    covariance_matrix = returns.cov().to_numpy(dtype=float)

    # 3. Use np.random.default_rng(random_seed) to create a seeded random generator.
    rng = np.random.default_rng(random_seed)

    # 4. Simulate correlated asset returns with shape
    #    (simulations, horizon_days, n_assets).
    simulated_returns = rng.multivariate_normal(
        mean=mean_vector,
        cov=covariance_matrix,
        size=(simulations, horizon_days),
    )

    # 5. Convert the simulated path into one cumulative return vector per simulation.
    if horizon_days == 1:
        simulated_asset_returns = np.squeeze(simulated_returns, axis=1)
    else:
        simulated_asset_returns = simulated_returns.sum(axis=1)

    # 6. Multiply simulated asset returns by weights and sum across assets.
    portfolio_simulations = simulated_asset_returns @ weight_array

    # 7. Return one portfolio return per simulation.
    return np.asarray(portfolio_simulations, dtype=float)
