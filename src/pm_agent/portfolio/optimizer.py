"""Portfolio allocation optimizer using Modern Portfolio Theory."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from pm_agent.schemas import Signal


def optimize_weights(
    signals: list[Signal],
    returns_cov_matrix: np.ndarray | None = None,
    max_weight: float = 0.2,
    min_weight: float = 0.0,
) -> np.ndarray:
    """
    Optimize position weights using mean-variance optimization.
    
    Args:
        signals: List of signals to allocate
        returns_cov_matrix: Covariance matrix of returns (if None, uses identity)
        max_weight: Maximum weight per position (default 20%)
        min_weight: Minimum weight per position (default 0%)
    
    Returns:
        Optimized weights array
    """
    n_assets = len(signals)

    if n_assets == 0:
        return np.array([])

    if n_assets == 1:
        return np.array([1.0])

    # Expected returns (use signal strength as proxy)
    expected_returns = np.array([s.strength for s in signals])

    # Use identity matrix if no covariance provided
    if returns_cov_matrix is None:
        returns_cov_matrix = np.eye(n_assets) * 0.01  # Default 1% volatility

    # Ensure covariance matrix is square and matches n_assets
    if returns_cov_matrix.shape != (n_assets, n_assets):
        returns_cov_matrix = np.eye(n_assets) * 0.01

    # Objective: maximize Sharpe ratio (minimize negative Sharpe)
    def neg_sharpe(weights: np.ndarray) -> float:
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(
            np.dot(weights.T, np.dot(returns_cov_matrix, weights))
        )
        if portfolio_vol == 0:
            return -portfolio_return * 100  # Penalize if no volatility
        return -portfolio_return / portfolio_vol

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Weights sum to 1
    ]

    # Bounds: each weight between min_weight and max_weight
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]

    # Initial guess: equal weights
    x0 = np.ones(n_assets) / n_assets

    try:
        result = minimize(
            neg_sharpe,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if result.success:
            return result.x
        else:
            # Fallback to equal weights if optimization fails
            return np.ones(n_assets) / n_assets

    except Exception:
        # Fallback to equal weights on error
        return np.ones(n_assets) / n_assets

