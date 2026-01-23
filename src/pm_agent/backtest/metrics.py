"""Enhanced performance metrics for backtesting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """
    Calmar ratio: CAGR / max drawdown.
    
    Higher is better. Measures return per unit of worst-case risk.
    """
    if max_drawdown == 0:
        return 0.0
    return float(cagr / abs(max_drawdown))


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Information ratio: (mean(returns - benchmark)) / std(returns - benchmark).
    
    Measures risk-adjusted excess return vs benchmark.
    """
    excess = returns - benchmark_returns
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std())


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega ratio: probability-weighted gains / probability-weighted losses.
    
    Higher is better. Measures the ratio of gains above threshold to losses below threshold.
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]

    if len(losses) == 0 or losses.sum() == 0:
        return float("inf") if len(gains) > 0 else 0.0

    return float(gains.sum() / losses.sum())


def tail_risk_metrics(returns: pd.Series) -> dict:
    """
    Calculate tail risk metrics: skewness and kurtosis.
    
    Returns:
        dict with 'skewness' and 'kurtosis'
    """
    from scipy import stats

    try:
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))
    except Exception:
        skew = 0.0
        kurt = 0.0

    return {
        "skewness": skew,
        "kurtosis": kurt,
        "tail_risk": "high" if abs(kurt) > 3 else "normal",
    }


def brier_score_decomposition(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Decompose Brier score into uncertainty, resolution, and calibration components.
    
    Brier = Uncertainty - Resolution + Calibration
    
    Returns:
        dict with 'brier', 'uncertainty', 'resolution', 'calibration'
    """
    # Overall Brier score
    brier = float(np.mean((y_pred - y_true) ** 2))

    # Uncertainty (variance of outcomes)
    uncertainty = float(np.var(y_true))

    # Resolution (variance of predictions)
    resolution = float(np.var(y_pred))

    # Calibration (reliability)
    # Bin predictions and calculate calibration
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    calibration = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            pred_mean = y_pred[mask].mean()
            obs_mean = y_true[mask].mean()
            calibration += mask.sum() * (pred_mean - obs_mean) ** 2

    calibration = float(calibration / len(y_pred))

    return {
        "brier": brier,
        "uncertainty": uncertainty,
        "resolution": resolution,
        "calibration": calibration,
        "decomposition_check": abs(brier - (uncertainty - resolution + calibration)) < 1e-6,
    }


def calculate_all_metrics(
    equity_curve: pd.DataFrame,
    trades: list,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        equity_curve: DataFrame with columns ['date', 'equity']
        trades: List of Trade objects
        benchmark_returns: Optional benchmark returns for comparison
    
    Returns:
        dict with all metrics
    """
    from pm_agent.backtest.engine import sharpe, max_drawdown, sortino

    if len(equity_curve) == 0:
        return {}

    # Basic returns
    returns = equity_curve["equity"].pct_change().dropna()

    # Basic metrics
    total_return = (equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0]) - 1.0
    max_dd = max_drawdown(equity_curve["equity"])
    sharpe_ratio = sharpe(returns)
    sortino_ratio = sortino(returns)

    # CAGR
    days = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days
    if days > 0:
        cagr = ((equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0]) ** (365.25 / days)) - 1.0
    else:
        cagr = 0.0

    metrics = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe_ratio),
        "sortino": float(sortino_ratio),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar_ratio(cagr, max_dd)),
    }

    # Trade-based metrics
    if trades:
        trade_returns = [t.pnl_pct for t in trades]
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0.0
        avg_loss = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        metrics.update(
            {
                "win_rate": float(win_rate),
                "total_trades": len(trades),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "profit_factor": float(profit_factor),
            }
        )

    # Tail risk
    tail_risk = tail_risk_metrics(returns)
    metrics.update(tail_risk)

    # Benchmark comparison
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        info_ratio = information_ratio(returns, benchmark_returns)
        metrics["information_ratio"] = float(info_ratio)

    # Omega ratio
    omega = omega_ratio(returns)
    metrics["omega_ratio"] = float(omega)

    return metrics

