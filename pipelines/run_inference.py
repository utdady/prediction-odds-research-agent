from __future__ import annotations

import asyncio

import pandas as pd

from pm_agent.config import settings
from pm_agent.db import get_session
from pm_agent.logging import configure_logging
from pm_agent.notifications.alerts import AlertManager
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


async def run_ensemble_strategy(session) -> int:
    """Generate signals using ensemble of strategies."""
    from pm_agent.strategies.ensemble import EnsembleStrategy

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
        return 0

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    ensemble = EnsembleStrategy()
    n = 0

    for _, r in df.iterrows():
        features = {
            "entity_id": r["entity_id"],
            "ts": r["ts"].to_pydatetime(),
            "p_now": float(r.get("p_now", 0.5)),
            "delta_p_1h": float(r.get("delta_p_1h", 0.0)),
            "delta_p_1d": float(r.get("delta_p_1d", 0.0)),
            "rolling_std_p_1d": float(r.get("rolling_std_p_1d", 0.0)),
            "liquidity_score": float(r.get("liquidity_score", 0.0)),
            "venue_disagreement": float(r.get("venue_disagreement", 0.0)),
            "time_to_resolution_days": float(r.get("time_to_resolution_days", 0.0)),
        }

        signal = ensemble.generate_signal(features)
        if signal:
            await upsert_signal(session, signal)
            n += 1

    return n


async def run() -> None:
    configure_logging()
    alerts = AlertManager(email=settings.alert_email, slack_webhook=settings.alert_slack_webhook)
    
    async with get_session() as session:
        n_rule = await run_rule_strategy(session)
        n_ml = await run_ml_strategy(session)
        n_ensemble = await run_ensemble_strategy(session)
        
        # Send alerts for high-confidence signals
        if n_rule > 0 or n_ml > 0 or n_ensemble > 0:
            signal_rows = await fetch_all(
                session,
                """
                SELECT entity_id, ts, strategy, side, strength, horizon_days, meta
                FROM signals
                WHERE ts > NOW() - INTERVAL '1 hour'
                ORDER BY strength DESC
                """,
            )
            for row in signal_rows:
                signal = Signal(
                    entity_id=row["entity_id"],
                    ts=row["ts"],
                    strategy=row["strategy"],
                    side=row["side"],
                    strength=float(row["strength"]),
                    horizon_days=int(row["horizon_days"]),
                    meta=row.get("meta", {}),
                )
                await alerts.send_signal_alert(signal)
        
        await session.commit()
        log.info("run_inference_done", n_rule=n_rule, n_ml=n_ml, n_ensemble=n_ensemble)


if __name__ == "__main__":
    asyncio.run(run())

