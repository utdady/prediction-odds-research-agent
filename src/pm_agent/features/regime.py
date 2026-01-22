"""Market regime detection (bull/bear/choppy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def detect_market_regime(spy_returns: pd.Series) -> str:
    """
    Classify current market state using Gaussian Mixture Model.
    
    Args:
        spy_returns: Series of SPY returns (daily)
    
    Returns:
        Regime string: 'bull', 'bear', or 'choppy'
    """
    if len(spy_returns) < 50:
        return "unknown"

    # Features for regime detection
    features = pd.DataFrame(
        {
            "returns": spy_returns,
            "volatility": spy_returns.rolling(20).std(),
            "trend": spy_returns.rolling(50).mean(),
        }
    ).dropna()

    if len(features) < 20:
        return "unknown"

    try:
        # Fit Gaussian Mixture Model (3 components: bull, bear, choppy)
        gmm = GaussianMixture(n_components=3, random_state=42, max_iter=100)
        gmm.fit(features.values)

        # Predict current regime
        current_state = gmm.predict(features.iloc[[-1]].values)[0]

        # Classify states based on mean returns
        means = gmm.means_[:, 0]  # Mean returns for each component
        sorted_indices = np.argsort(means)

        # Highest mean = bull, lowest = bear, middle = choppy
        regime_map = {
            sorted_indices[2]: "bull",  # Highest returns
            sorted_indices[0]: "bear",  # Lowest returns
            sorted_indices[1]: "choppy",  # Middle returns
        }

        return regime_map.get(current_state, "unknown")
    except Exception:
        return "unknown"


def get_regime_adaptive_threshold(regime: str, base_threshold: float = 0.08) -> float:
    """
    Adjust signal threshold based on market regime.
    
    Args:
        regime: Market regime ('bull', 'bear', 'choppy')
        base_threshold: Base threshold for signal generation
    
    Returns:
        Adjusted threshold
    """
    if regime == "bull":
        return base_threshold * 0.6  # Lower threshold (easier to trigger)
    elif regime == "bear":
        return base_threshold * 1.5  # Higher threshold (be selective)
    else:  # choppy or unknown
        return base_threshold * 2.0  # Much higher threshold (avoid trading)
