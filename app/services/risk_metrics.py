from __future__ import annotations

import numpy as np


def compute_var(simulated_returns: np.ndarray, confidence_level: float) -> float:
    """Compute Value at Risk from simulated portfolio returns.

    Args:
        simulated_returns: One-dimensional array of simulated portfolio returns.
        confidence_level: Confidence level used to define the tail cutoff.

    Returns:
        A positive float representing the loss magnitude at the requested
        confidence level.
    """

    percentile_level = (1.0 - confidence_level) * 100.0
    var_threshold = float(np.percentile(simulated_returns, percentile_level))
    return max(0.0, -var_threshold)


def compute_es(simulated_returns: np.ndarray, confidence_level: float) -> float:
    """Compute Expected Shortfall from simulated portfolio returns.

    Args:
        simulated_returns: One-dimensional array of simulated portfolio returns.
        confidence_level: Confidence level used to define the tail cutoff.

    Returns:
        A positive float representing the average loss magnitude in the tail.

    Raises:
        ValueError: If no simulated returns fall below the VaR threshold.
    """

    percentile_level = (1.0 - confidence_level) * 100.0
    var_threshold = float(np.percentile(simulated_returns, percentile_level))
    tail_returns = simulated_returns[simulated_returns < var_threshold]

    if tail_returns.size == 0:
        raise ValueError(
            "No simulated returns fell below the VaR threshold when computing ES."
        )

    return max(0.0, -float(tail_returns.mean()))
