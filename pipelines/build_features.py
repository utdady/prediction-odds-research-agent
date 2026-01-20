from __future__ import annotations

import asyncio

import numpy as np
from datetime import datetime, timedelta, timezone
import pandas as pd

from pm_agent.db import get_session
from pm_agent.logging import configure_logging
import structlog

from pm_agent.market_mapping import load_mapping
from pm_agent.repo.upserts import upsert_features
from pm_agent.sql import execute, fetch_all
from pm_agent.schemas import FeatureRow


log = structlog.get_logger(__name__)


async def ensure_entities_and_maps(session) -> None:
    mapping = load_mapping()
    if not mapping:
        return

    # upsert entities + maps
    for m in mapping:
        ticker = m["entity_ticker"]
        entity_id = ticker
        await execute(
            session,
            """
            INSERT INTO entities(entity_id, ticker, name)
            VALUES (:entity_id, :ticker, :name)
            ON CONFLICT (entity_id) DO UPDATE SET ticker=EXCLUDED.ticker
            """,
            {"entity_id": entity_id, "ticker": ticker, "name": ticker},
        )
        await execute(
            session,
            """
            INSERT INTO market_entity_map(market_id, entity_id, relationship_type, confidence, notes)
            VALUES (:market_id, :entity_id, :relationship_type, :confidence, :notes)
            ON CONFLICT (market_id, entity_id, relationship_type) DO UPDATE SET
              confidence=EXCLUDED.confidence,
              notes=EXCLUDED.notes
            """,
            {
                "market_id": m["market_id"],
                "entity_id": entity_id,
                "relationship_type": m["relationship_type"],
                "confidence": float(m.get("confidence", 1.0)),
                "notes": m.get("notes"),
            },
        )


async def run() -> None:
    configure_logging()
    async with get_session() as session:
        await ensure_entities_and_maps(session)
        await session.commit()

        rows = await fetch_all(
            session,
            """
            SELECT m.entity_id, t.tick_ts, t.p_norm, t.volume, t.venue_id, mk.resolution_ts
            FROM market_entity_map m
            JOIN odds_ticks t ON t.market_id = m.market_id
            JOIN markets mk ON mk.market_id = m.market_id
            ORDER BY m.entity_id, t.tick_ts
            """,
        )
        if not rows:
            log.info("build_features_no_rows")
            return

        df = pd.DataFrame(rows)
        df["tick_ts"] = pd.to_datetime(df["tick_ts"], utc=True)
        if df["resolution_ts"].notna().any():
            df["resolution_ts"] = pd.to_datetime(df["resolution_ts"], utc=True)
        else:
            df["resolution_ts"] = None

        def get_previous_market_close(snapshot_time: datetime) -> datetime:
            """Get most recent US market close (16:00 ET ~ 21:00 UTC) before snapshot_time."""
            snapshot_utc = snapshot_time.astimezone(timezone.utc)
            close = snapshot_utc.replace(hour=21, minute=0, second=0, microsecond=0)
            if snapshot_utc < close:
                close -= timedelta(days=1)
            while close.weekday() >= 5:  # Saturday=5, Sunday=6
                close -= timedelta(days=1)
            return close

        out_rows: list[FeatureRow] = []
        for entity_id, g in df.groupby("entity_id"):
            g = g.sort_values("tick_ts")

            # choose snapshot grid using unique tick dates for this entity
            tick_dates = g["tick_ts"].dt.normalize().unique()

            for day in tick_dates:
                # use 06:00 UTC snapshots as “end of day” feature time
                snapshot_time = (day + pd.Timedelta(hours=6)).to_pydatetime().replace(tzinfo=timezone.utc)
                cutoff = get_previous_market_close(snapshot_time)

                valid = g[g["tick_ts"] <= cutoff]
                if valid.empty:
                    continue

                agg = valid.groupby("tick_ts").agg(
                    p_now=("p_norm", "mean"),
                    vol=("volume", "mean"),
                )
                agg["delta_p_1h"] = agg["p_now"].diff()
                agg["delta_p_1d"] = agg["p_now"].diff()
                agg["rolling_std_p_1d"] = agg["p_now"].rolling(window=3, min_periods=1).std()
                agg["liquidity_score"] = np.clip((agg["vol"].fillna(0) / (agg["vol"].fillna(0).max() + 1e-9)), 0, 1)

                # disagreement (kalshi - poly) at same ts if both exist
                piv = valid.pivot_table(index="tick_ts", columns="venue_id", values="p_norm", aggfunc="last")
                agg["venue_disagreement"] = piv.get("kalshi", pd.Series()) - piv.get("polymarket", pd.Series())
                agg["venue_disagreement"] = agg["venue_disagreement"].fillna(0.0)

                # time to resolution from first mapped market resolution (mock)
                res_ts = valid["resolution_ts"].dropna().iloc[0] if valid["resolution_ts"].notna().any() else None
                if res_ts is not None and pd.notna(res_ts):
                    agg["time_to_resolution_days"] = (res_ts - agg.index) / np.timedelta64(1, "D")
                else:
                    agg["time_to_resolution_days"] = np.nan

                for ts, r in agg.iterrows():
                    out_rows.append(
                        FeatureRow(
                            entity_id=entity_id,
                            ts=ts.to_pydatetime(),
                            p_now=None if pd.isna(r["p_now"]) else float(r["p_now"]),
                            delta_p_1h=None if pd.isna(r["delta_p_1h"]) else float(r["delta_p_1h"]),
                            delta_p_1d=None if pd.isna(r["delta_p_1d"]) else float(r["delta_p_1d"]),
                            rolling_std_p_1d=None if pd.isna(r["rolling_std_p_1d"]) else float(r["rolling_std_p_1d"]),
                            liquidity_score=None if pd.isna(r["liquidity_score"]) else float(r["liquidity_score"]),
                            venue_disagreement=None if pd.isna(r["venue_disagreement"]) else float(r["venue_disagreement"]),
                            time_to_resolution_days=None if pd.isna(r["time_to_resolution_days"]) else float(r["time_to_resolution_days"]),
                            raw={},
                        )
                    )

        await upsert_features(session, out_rows)
        await session.commit()
        log.info("build_features_done", n=len(out_rows))


if __name__ == "__main__":
    asyncio.run(run())

