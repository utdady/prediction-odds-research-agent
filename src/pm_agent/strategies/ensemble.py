"""Ensemble strategy combining multiple approaches."""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pm_agent.schemas import Signal

log = structlog.get_logger(__name__)


class RuleStrategyV1:
    """Simple rule-based strategy."""

    def generate_signal(self, features: dict) -> Signal | None:
        """Generate signal based on rules."""
        delta_p = features.get("delta_p_1d", 0.0)
        liquidity = features.get("liquidity_score", 0.0)

        if liquidity < 0.2 or delta_p < 0.08:
            return None

        from pm_agent.schemas import Signal
        from datetime import datetime

        return Signal(
            entity_id=features["entity_id"],
            ts=features["ts"],
            strategy="RuleStrategyV1",
            side="LONG",
            strength=min(delta_p, 1.0),
            horizon_days=5,
            meta={"rule": "delta_p_1d"},
        )


class MomentumStrategy:
    """Momentum-based strategy."""

    def generate_signal(self, features: dict) -> Signal | None:
        """Generate signal based on momentum."""
        delta_p = features.get("delta_p_1d", 0.0)
        momentum = features.get("momentum_1d", 0.0) if "momentum_1d" in features else delta_p

        if momentum < 0.05:
            return None

        from pm_agent.schemas import Signal
        from datetime import datetime

        return Signal(
            entity_id=features["entity_id"],
            ts=features["ts"],
            strategy="MomentumStrategy",
            side="LONG",
            strength=min(abs(momentum), 1.0),
            horizon_days=3,
            meta={"momentum": momentum},
        )


class MeanReversionStrategy:
    """Mean reversion strategy."""

    def generate_signal(self, features: dict) -> Signal | None:
        """Generate signal when price deviates from mean."""
        p_now = features.get("p_now", 0.5)
        delta_p = features.get("delta_p_1d", 0.0)
        rolling_std = features.get("rolling_std_p_1d", 0.0)

        # Look for oversold conditions (low probability, negative momentum)
        if p_now < 0.3 and delta_p < -0.05 and rolling_std > 0.1:
            from pm_agent.schemas import Signal
            from datetime import datetime

            return Signal(
                entity_id=features["entity_id"],
                ts=features["ts"],
                strategy="MeanReversionStrategy",
                side="LONG",
                strength=min((0.3 - p_now) / 0.3, 1.0),
                horizon_days=7,
                meta={"mean_reversion": True},
            )

        return None


class ModelStrategyV1:
    """ML model-based strategy."""

    def __init__(self, model_path: str = "artifacts/model_v1.joblib"):
        self.model_path = model_path
        self.model = None
        self.calibrator = None
        self.feature_cols = None
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model."""
        from pathlib import Path
        import joblib

        if not Path(self.model_path).exists():
            log.warning("model_not_loaded", path=self.model_path)
            return

        try:
            artifacts = joblib.load(self.model_path)
            self.model = artifacts["model"]
            self.calibrator = artifacts["calibrator"]
            self.feature_cols = artifacts["features"]
        except Exception as e:
            log.warning("model_load_failed", error=str(e))

    def generate_signal(self, features: dict) -> Signal | None:
        """Generate signal using ML model."""
        if self.model is None:
            return None

        from pm_agent.config import settings
        import numpy as np

        # Extract features
        X = np.array([[features.get(col, 0.0) for col in self.feature_cols]])

        try:
            prob_raw = self.model.predict_proba(X)[0, 1]
            prob_cal = self.calibrator.transform([prob_raw])[0] if self.calibrator else prob_raw

            if prob_cal < settings.ml_confidence_threshold:
                return None

            from pm_agent.schemas import Signal

            return Signal(
                entity_id=features["entity_id"],
                ts=features["ts"],
                strategy="ModelV1",
                side="LONG",
                strength=float(prob_cal),
                horizon_days=settings.holding_period_days,
                meta={"prob_calibrated": float(prob_cal)},
            )
        except Exception as e:
            log.warning("model_prediction_failed", error=str(e))
            return None


class EnsembleStrategy:
    """Combines multiple strategies with weighted voting."""

    def __init__(self):
        self.strategies = {
            "rule": RuleStrategyV1(),
            "ml": ModelStrategyV1(),
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
        }

        # Strategy weights (can be learned via meta-optimization)
        self.weights = {
            "rule": 0.3,
            "ml": 0.4,
            "momentum": 0.2,
            "mean_reversion": 0.1,
        }

    def generate_signal(self, features: dict) -> Signal | None:
        """Generate signal using ensemble of strategies."""
        signals = {}
        for name, strategy in self.strategies.items():
            try:
                signals[name] = strategy.generate_signal(features)
            except Exception as e:
                log.warning("strategy_failed", strategy=name, error=str(e))
                signals[name] = None

        # Weighted vote
        weighted_strength = sum(
            self.weights[name] * (sig.strength if sig else 0.0) for name, sig in signals.items()
        )

        # Count votes by side
        num_long = sum(1 for s in signals.values() if s and s.side == "LONG")
        num_short = sum(1 for s in signals.values() if s and s.side == "SHORT")

        # Consensus required (majority of strategies agree)
        if num_long >= 2 and weighted_strength > 0.5:
            from pm_agent.schemas import Signal

            return Signal(
                entity_id=features["entity_id"],
                ts=features["ts"],
                strategy="Ensemble",
                side="LONG",
                strength=min(weighted_strength, 1.0),
                horizon_days=int(
                    sum(
                        self.weights[name] * (sig.horizon_days if sig else 5)
                        for name, sig in signals.items()
                    )
                ),
                meta={
                    "ensemble_votes": {
                        name: sig.strength if sig else None for name, sig in signals.items()
                    },
                    "weighted_strength": weighted_strength,
                },
            )
        elif num_short >= 2 and weighted_strength < -0.5:
            from pm_agent.schemas import Signal

            return Signal(
                entity_id=features["entity_id"],
                ts=features["ts"],
                strategy="Ensemble",
                side="SHORT",
                strength=min(abs(weighted_strength), 1.0),
                horizon_days=5,
                meta={"ensemble_votes": {name: sig.strength if sig else None for name, sig in signals.items()}},
            )

        return None  # No consensus
