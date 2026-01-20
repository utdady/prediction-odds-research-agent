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


async def run() -> None:
    configure_logging()
    async with get_session() as session:
        n = await run_rule_strategy(session)
        await session.commit()
        log.info("run_inference_done", n=n)


if __name__ == "__main__":
    asyncio.run(run())

