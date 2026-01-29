from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from pm_agent.config import settings
from pm_agent.db import get_session
from pm_agent.logging import configure_logging
from pm_agent.prices.provider import LocalCSVPriceProvider
from pm_agent.sql import fetch_all

import structlog


log = structlog.get_logger(__name__)


def make_labels(features: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    price = LocalCSVPriceProvider()
    labels = []
    for _, row in features.iterrows():
        ticker = row["entity_id"]
        # row["ts"] may already be tz-aware (we set df["ts"] = to_datetime(..., utc=True)).
        ts = pd.Timestamp(row["ts"])
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        ts = ts.normalize()
        px = price.load_prices(ticker)
        spy = price.load_prices("SPY")

        if ts not in px.index or ts not in spy.index:
            labels.append(np.nan)
            continue

        idx = list(px.index)
        i = idx.index(ts)
        j = min(i + horizon_days, len(idx) - 1)
        px_ret = np.log(px.iloc[j]["close"] / px.iloc[i]["open"])

        idx2 = list(spy.index)
        i2 = idx2.index(ts)
        j2 = min(i2 + horizon_days, len(idx2) - 1)
        spy_ret = np.log(spy.iloc[j2]["close"] / spy.iloc[i2]["open"])

        labels.append(1.0 if (px_ret - spy_ret) > 0 else 0.0)
    return pd.Series(labels, index=features.index)


async def run() -> None:
    configure_logging()
    async with get_session() as session:
        rows = await fetch_all(
            session,
            """
            SELECT entity_id, ts, p_now, delta_p_1h, delta_p_1d, rolling_std_p_1d,
                   liquidity_score, venue_disagreement, time_to_resolution_days
            FROM features
            ORDER BY ts
            """,
        )

    if not rows:
        log.info("train_model_no_features")
        return

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    y = make_labels(df, horizon_days=settings.holding_period_days)
    df = df.assign(label=y).dropna()

    if len(df) < 100:
        log.warning("train_model_too_few_rows", n=len(df), min_required=100)
        return

    X = df[["p_now", "delta_p_1h", "delta_p_1d", "rolling_std_p_1d", "liquidity_score", "venue_disagreement", "time_to_resolution_days"]].fillna(0.0)
    y = df["label"].astype(int)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    p_cal = iso.transform(p)

    brier = float(np.mean((p_cal - y.values) ** 2))

    # persist as artifacts
    import joblib
    from pathlib import Path

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump({"model": clf, "calibrator": iso, "features": list(X.columns)}, "artifacts/model_v1.joblib")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': list(X.columns),
        'importance': clf.coef_[0] if hasattr(clf, 'coef_') else [0.0] * len(X.columns)
    }).sort_values('importance', key=abs, ascending=False)
    feature_importance.to_csv("artifacts/feature_importance.csv", index=False)

    # MLflow (optional)
    if settings.mlflow_tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment)
            with mlflow.start_run(run_name="train_model_v1"):
                mlflow.log_param("model", "LogisticRegression")
                mlflow.log_param("calibration", "Isotonic")
                mlflow.log_metric("brier", brier)
                mlflow.log_artifact("artifacts/model_v1.joblib")
        except Exception as e:
            log.warning("mlflow_logging_failed", error=str(e))

    log.info("train_model_done", brier=brier, n=len(df))


if __name__ == "__main__":
    asyncio.run(run())

