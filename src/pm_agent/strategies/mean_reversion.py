"""Mean reversion strategy."""
from __future__ import annotations

from pm_agent.schemas import Signal
from pm_agent.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy - buy when price drops below mean."""

    def __init__(self, threshold: float = -0.10):
        self.threshold = threshold

    def generate_signal(self, features: dict) -> Signal | None:
        """Generate signal based on mean reversion."""
        delta_p_1d = features.get("delta_p_1d")
        rolling_std = features.get("rolling_std_p_1d")
        p_now = features.get("p_now")

        if delta_p_1d is None or rolling_std is None or p_now is None:
            return None

        # Price dropped significantly (potential oversold)
        if delta_p_1d < self.threshold and rolling_std > 0.01:
            # Strength based on how far below mean
            strength = abs(delta_p_1d) / 0.20  # Normalize to 0-1
            strength = min(strength, 0.8)  # Cap at 80%

            return Signal(
                entity_id=features["entity_id"],
                ts=features["ts"],
                strategy="MeanReversion",
                side="LONG",
                strength=float(strength),
                horizon_days=3,  # Shorter horizon for mean reversion
                meta={"delta_p": delta_p_1d, "std": rolling_std},
            )

        return None

    def get_name(self) -> str:
        return "MeanReversion"

