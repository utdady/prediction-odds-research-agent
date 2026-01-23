"""Meta-learning for optimizing ensemble weights."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if TYPE_CHECKING:
    from pm_agent.schemas import Signal


def optimize_ensemble_weights(
    historical_signals: dict[str, list[Signal]],
    historical_returns: pd.Series,
    min_weight: float = 0.05,
    max_weight: float = 0.6,
) -> dict[str, float]:
    """
    Learn optimal ensemble weights via optimization.
    
    Args:
        historical_signals: Dict mapping strategy names to lists of signals
        historical_returns: Series of returns (aligned with signals)
        min_weight: Minimum weight per strategy
        max_weight: Maximum weight per strategy
    
    Returns:
        Dict mapping strategy names to optimized weights
    """
    strategy_names = list(historical_signals.keys())
    n_strategies = len(strategy_names)

    if n_strategies == 0:
        return {}

    # Convert signals to prediction matrix
    # Each row is a time period, each column is a strategy's prediction
    predictions = np.zeros((len(historical_returns), n_strategies))

    for i, strategy_name in enumerate(strategy_names):
        signals = historical_signals[strategy_name]
        # Map signals to time periods (simplified - would need proper alignment)
        for j, signal in enumerate(signals[: len(historical_returns)]):
            if j < len(predictions):
                predictions[j, i] = signal.strength if signal.side == "LONG" else -signal.strength

    # Objective: maximize Sharpe ratio of weighted ensemble
    def objective(weights: np.ndarray) -> float:
        # Normalize weights to sum to 1
        weights = weights / weights.sum()

        # Weighted predictions
        ensemble_pred = predictions @ weights

        # Generate returns based on predictions (simplified)
        # Positive prediction -> positive return, negative -> negative
        simulated_returns = ensemble_pred * historical_returns.abs()

        # Sharpe ratio (negative because we minimize)
        if simulated_returns.std() == 0:
            return 1e6  # Penalty for zero volatility

        sharpe = simulated_returns.mean() / simulated_returns.std()
        return -sharpe  # Minimize negative Sharpe = maximize Sharpe

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    # Bounds: each weight between min and max
    bounds = [(min_weight, max_weight) for _ in range(n_strategies)]

    # Initial guess: equal weights
    x0 = np.ones(n_strategies) / n_strategies

    # Optimize
    try:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if result.success:
            optimal_weights = result.x / result.x.sum()  # Normalize
            return dict(zip(strategy_names, optimal_weights))
        else:
            # Fallback to equal weights
            return {name: 1.0 / n_strategies for name in strategy_names}
    except Exception:
        # Fallback to equal weights on error
        return {name: 1.0 / n_strategies for name in strategy_names}


def optimize_weights_cross_validation(
    historical_signals: dict[str, list[Signal]],
    historical_returns: pd.Series,
    n_folds: int = 5,
) -> dict[str, float]:
    """
    Optimize weights using k-fold cross-validation.
    
    Args:
        historical_signals: Dict mapping strategy names to lists of signals
        historical_returns: Series of returns
        n_folds: Number of cross-validation folds
    
    Returns:
        Dict mapping strategy names to optimized weights
    """
    n = len(historical_returns)
    fold_size = n // n_folds

    all_weights = []

    for fold in range(n_folds):
        # Split into train/validation
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

        # Use remaining data for training
        train_returns = pd.concat(
            [
                historical_returns[:val_start],
                historical_returns[val_end:],
            ]
        )

        # Optimize on training set
        fold_weights = optimize_ensemble_weights(historical_signals, train_returns)
        if fold_weights:
            all_weights.append(fold_weights)

    if not all_weights:
        return {}

    # Average weights across folds
    strategy_names = list(all_weights[0].keys())
    avg_weights = {
        name: np.mean([w[name] for w in all_weights]) for name in strategy_names
    }

    # Normalize to sum to 1
    total = sum(avg_weights.values())
    return {name: w / total for name, w in avg_weights.items()}

