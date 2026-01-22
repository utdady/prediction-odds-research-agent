"""Multi-timeframe feature computation."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


TIMEFRAMES = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "7d": timedelta(days=7),
}


def compute_multitf_features(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features at multiple timeframes.
    
    Args:
        ticks: DataFrame with columns ['tick_ts', 'p_norm', 'volume']
    
    Returns:
        DataFrame with multi-timeframe features
    """
    if ticks.empty:
        return pd.DataFrame()

    ticks = ticks.copy()
    ticks = ticks.set_index("tick_ts").sort_index()

    features_list = []

    for tf_name, window in TIMEFRAMES.items():
        # Resample ticks to timeframe
        resampled = ticks.resample(window).agg(
            {
                "p_norm": ["mean", "std", "min", "max"],
                "volume": "sum",
            }
        )

        if resampled.empty:
            continue

        # Flatten column names
        resampled.columns = [f"{col[0]}_{tf_name}" for col in resampled.columns]

        # Compute momentum
        resampled[f"momentum_{tf_name}"] = resampled[f"mean_{tf_name}"].pct_change()

        # Compute trend strength (linear regression slope over rolling window)
        def _trend_slope(x):
            if len(x) < 2:
                return 0.0
            try:
                return np.polyfit(range(len(x)), x.values, 1)[0]
            except Exception:
                return 0.0

        resampled[f"trend_{tf_name}"] = (
            resampled[f"mean_{tf_name}"].rolling(window=min(5, len(resampled)), min_periods=2).apply(_trend_slope, raw=False)
        )

        features_list.append(resampled)

    if not features_list:
        return pd.DataFrame()

    # Combine all timeframes
    result = pd.concat(features_list, axis=1)

    return result

