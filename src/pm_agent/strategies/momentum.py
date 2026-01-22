"""Momentum-based strategy."""
from __future__ import annotations

from pm_agent.schemas import Signal
from pm_agent.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Momentum strategy based on price changes."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def generate_signal(self, features: dict) -> Signal | None:
        """Generate signal based on momentum."""
        delta_p_1d = features.get("delta_p_1d")
        delta_p_1h = features.get("delta_p_1h")

        if delta_p_1d is None:
            return None

        # Strong momentum: both 1h and 1d positive
        if delta_p_1h and delta_p_1h > 0 and delta_p_1d > self.threshold:
            strength = min(delta_p_1d, 0.15)  # Cap at 15%
            return Signal(
                entity_id=features["entity_id"],
                ts=features["ts"],
                strategy="Momentum",
                side="LONG",
                strength=float(strength),
                horizon_days=5,
                meta={"momentum_1h": delta_p_1h, "momentum_1d": delta_p_1d},
            )

        return None

    def get_name(self) -> str:
        return "Momentum"

