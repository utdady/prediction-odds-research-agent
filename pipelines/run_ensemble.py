"""Run ensemble strategy inference."""
from __future__ import annotations

import asyncio

import pandas as pd

from pm_agent.config import settings
from pm_agent.db import get_session
from pm_agent.logging import configure_logging
from pm_agent.repo.upserts import upsert_signal
from pm_agent.strategies.ensemble import EnsembleStrategy
from pm_agent.sql import fetch_all

import structlog


log = structlog.get_logger(__name__)


async def run() -> None:
    """Run ensemble strategy to generate signals."""
    configure_logging()
    
    ensemble = EnsembleStrategy()
    
    async with get_session() as session:
        # Fetch features
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
            log.info("ensemble_no_features")
            return
        
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        
        n_signals = 0
        for _, row in df.iterrows():
            features = {
                "entity_id": row["entity_id"],
                "ts": row["ts"].to_pydatetime(),
                "p_now": row.get("p_now"),
                "delta_p_1h": row.get("delta_p_1h"),
                "delta_p_1d": row.get("delta_p_1d"),
                "rolling_std_p_1d": row.get("rolling_std_p_1d"),
                "liquidity_score": row.get("liquidity_score"),
                "venue_disagreement": row.get("venue_disagreement"),
                "time_to_resolution_days": row.get("time_to_resolution_days"),
                "horizon_days": settings.holding_period_days,
            }
            
            signal = ensemble.generate_signal(features)
            
            if signal:
                await upsert_signal(session, signal)
                n_signals += 1
        
        await session.commit()
        log.info("ensemble_inference_done", n_signals=n_signals)


if __name__ == "__main__":
    asyncio.run(run())

