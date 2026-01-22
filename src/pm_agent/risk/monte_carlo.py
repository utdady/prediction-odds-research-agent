"""Monte Carlo simulation for risk analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pm_agent.backtest.engine import run_event_driven_backtest, BacktestConfig, CostModel


def run_monte_carlo_simulation(
    signals: pd.DataFrame,
    config: BacktestConfig,
    historical_returns: pd.Series | None = None,
    n_simulations: int = 10000,
) -> dict:
    """
    Simulate strategy under different market scenarios.
    
    Args:
        signals: DataFrame with columns [entity_id, ts, horizon_days]
        config: Backtest configuration
        historical_returns: Historical returns to sample from (if None, uses backtest results)
        n_simulations: Number of Monte Carlo simulations
    
    Returns:
        Dictionary with risk metrics
    """
    # First, run base backtest to get historical returns
    if historical_returns is None:
        curve, trades = run_event_driven_backtest(signals, config)
        if len(curve) > 0:
            historical_returns = curve["equity"].pct_change().dropna()
        else:
            historical_returns = pd.Series([0.0] * 252)  # Default: flat returns

    results = []

    for _ in range(n_simulations):
        # Randomly sample from historical returns
        simulated_returns = np.random.choice(
            historical_returns.values,
            size=252,  # 1 year of trading days
            replace=True,
        )

        # Simulate equity curve
        equity = 1.0
        for ret in simulated_returns:
            equity *= 1.0 + ret

        results.append(equity)

    results = np.array(results)

    # Calculate risk metrics
    value_at_risk_95 = np.percentile(results, 5)  # 95% VaR (5th percentile)
    value_at_risk_99 = np.percentile(results, 1)  # 99% VaR (1st percentile)
    expected_shortfall = results[results <= value_at_risk_95].mean()

    prob_profit = (results > 1.0).mean()
    prob_loss = (results < 1.0).mean()

    return {
        "mean_return": float(results.mean() - 1.0),
        "std_return": float(results.std()),
        "median_return": float(np.median(results) - 1.0),
        "var_95": float(value_at_risk_95 - 1.0),
        "var_99": float(value_at_risk_99 - 1.0),
        "expected_shortfall": float(expected_shortfall - 1.0),
        "prob_profit": float(prob_profit),
        "prob_loss": float(prob_loss),
        "min_return": float(results.min() - 1.0),
        "max_return": float(results.max() - 1.0),
        "simulation_results": results,
    }
