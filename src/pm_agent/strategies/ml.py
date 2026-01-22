"""ML-based strategy implementation."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from pm_agent.config import settings
from pm_agent.schemas import Signal
from pm_agent.strategies.base import BaseStrategy


class ModelStrategyV1(BaseStrategy):
    """ML model-based strategy."""

    def __init__(self):
        self.model = None
        self.calibrator = None
        self.feature_cols = None
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model from artifacts."""
        model_path = Path("artifacts/model_v1.joblib")
        if not model_path.exists():
            return

        artifacts = joblib.load(model_path)
        self.model = artifacts["model"]
        self.calibrator = artifacts["calibrator"]
        self.feature_cols = artifacts["features"]

    def generate_signal(self, features: dict) -> Signal | None:
        """Generate signal using ML model."""
        if self.model is None or self.calibrator is None:
            return None

        # Extract features in correct order
        X = pd.DataFrame([{
            col: features.get(col, 0.0)
            for col in self.feature_cols
        }])

        X = X.fillna(0.0)

        # Predict
        prob_raw = self.model.predict_proba(X)[0, 1]
        prob_cal = self.calibrator.transform([prob_raw])[0]

        if prob_cal < settings.ml_confidence_threshold:
            return None

        return Signal(
            entity_id=features["entity_id"],
            ts=features["ts"],
            strategy="ModelV1",
            side="LONG",
            strength=float(prob_cal),
            horizon_days=settings.holding_period_days,
            meta={"prob_calibrated": float(prob_cal)},
        )

    def get_name(self) -> str:
        return "ModelV1"

