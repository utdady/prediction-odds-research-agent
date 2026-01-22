"""Rule-based strategy implementation."""
from __future__ import annotations

from pm_agent.config import settings
from pm_agent.schemas import Signal
from pm_agent.strategies.base import BaseStrategy


class RuleStrategyV1(BaseStrategy):
    """Simple rule-based strategy using delta_p_1d threshold."""

    def generate_signal(self, features: dict) -> Signal | None:
        """Generate signal based on delta_p_1d and liquidity."""
        delta_p = features.get("delta_p_1d")
        liquidity = features.get("liquidity_score")

        if delta_p is None or liquidity is None:
            return None

        if liquidity < settings.rule_min_liquidity:
            return None

        if delta_p > settings.rule_delta_p_1d_threshold:
            return Signal(
                entity_id=features["entity_id"],
                ts=features["ts"],
                strategy="RuleStrategyV1",
                side="LONG",
                strength=float(delta_p),
                horizon_days=settings.holding_period_days,
                meta={"rule": "delta_p_1d"},
            )

        return None

    def get_name(self) -> str:
        return "RuleStrategyV1"

