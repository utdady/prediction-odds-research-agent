from __future__ import annotations

import asyncio

import pandas as pd

from pm_agent.config import settings
from pm_agent.db import get_session
from pm_agent.logging import configure_logging
from pm_agent.repo.upserts import upsert_signal
from pm_agent.schemas import Signal
from pm_agent.sql import fetch_all

import structlog


log = structlog.get_logger(__name__)


async def run_rule_strategy(session) -> int:
    rows = await fetch_all(
        session,
        """
        SELECT entity_id, ts, delta_p_1d, liquidity_score
        FROM features
        ORDER BY ts
        """,
    )
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    n = 0
    for _, r in df.iterrows():
        dp = r["delta_p_1d"]
        liq = r["liquidity_score"]
        if pd.isna(dp) or pd.isna(liq):
            continue
        if float(liq) < settings.rule_min_liquidity:
            continue
        if float(dp) > settings.rule_delta_p_1d_threshold:
            s = Signal(
                entity_id=r["entity_id"],
                ts=r["ts"].to_pydatetime(),
                strategy="RuleStrategyV1",
                side="LONG",
                strength=float(dp),
                horizon_days=settings.holding_period_days,
                meta={"rule": "delta_p_1d"},
            )
            await upsert_signal(session, s)
            n += 1

    return n


async def run_ml_strategy(session) -> int:
    """Generate signals using trained ML model."""
    from pathlib import Path
    import joblib

    model_path = Path("artifacts/model_v1.joblib")
    if not model_path.exists():
        log.warning("ml_model_not_found", path=str(model_path))
        return 0

    artifacts = joblib.load(model_path)
    model = artifacts["model"]
    calibrator = artifacts["calibrator"]
    feature_cols = artifacts["features"]

    rows = await fetch_all(session, "SELECT * FROM features ORDER BY ts")
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    X = df[feature_cols].fillna(0.0)
    probs_raw = model.predict_proba(X)[:, 1]
    probs_cal = calibrator.transform(probs_raw)

    n = 0
    for i, (_, r) in enumerate(df.iterrows()):
        if probs_cal[i] < settings.ml_confidence_threshold:
            continue
        s = Signal(
            entity_id=r["entity_id"],
            ts=r["ts"].to_pydatetime(),
            strategy="ModelV1",
            side="LONG",
            strength=float(probs_cal[i]),
            horizon_days=settings.holding_period_days,
            meta={"prob_calibrated": float(probs_cal[i])},
        )
        await upsert_signal(session, s)
        n += 1

    return n


async def run() -> None:
    configure_logging()
    async with get_session() as session:
        n_rule = await run_rule_strategy(session)
        n_ml = await run_ml_strategy(session)
        await session.commit()
        log.info("run_inference_done", n_rule=n_rule, n_ml=n_ml)


if __name__ == "__main__":
    asyncio.run(run())

