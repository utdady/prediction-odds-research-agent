from __future__ import annotations

import asyncio
import json

from pm_agent.config import settings
from pm_agent.connectors.mock import MockConnector
from pm_agent.db import get_session
from pm_agent.logging import configure_logging
import structlog

from pm_agent.data_quality.validators import validate_probability, validate_tick_ts
from pm_agent.repo.upserts import upsert_tick
from pm_agent.sql import execute, fetch_all


log = structlog.get_logger(__name__)


async def run() -> None:
    configure_logging()
    async with get_session() as session:
        market_rows = await fetch_all(session, "SELECT market_id, venue_id FROM markets")
        ids_by_venue: dict[str, list[str]] = {}
        for r in market_rows:
            ids_by_venue.setdefault(r["venue_id"], []).append(r["market_id"])

        kalshi = MockConnector("kalshi", "data/mock/kalshi_markets.json", "data/mock/kalshi_ticks.json")
        poly = MockConnector("polymarket", "data/mock/poly_markets.json", "data/mock/poly_ticks.json")

        ticks = []
        if "kalshi" in ids_by_venue:
            ticks += await kalshi.fetch_ticks(ids_by_venue["kalshi"])
        if "polymarket" in ids_by_venue:
            ticks += await poly.fetch_ticks(ids_by_venue["polymarket"])

        for t in ticks:
            ok_p, err_p = validate_probability(t.p_norm)
            ok_ts, err_ts = validate_tick_ts(t.tick_ts)

            if not ok_p:
                await execute(
                    session,
                    """
                    INSERT INTO data_quality_log(scope, level, message, context)
                    VALUES (:scope, :level, :message, :context::jsonb)
                    """,
                    {
                        "scope": "ingest_ticks",
                        "level": "error",
                        "message": "probability_range",
                        "context": json.dumps({"market_id": t.market_id, "venue_id": t.venue_id, "error": err_p}),
                    },
                )
                continue

            if not ok_ts:
                await execute(
                    session,
                    """
                    INSERT INTO data_quality_log(scope, level, message, context)
                    VALUES (:scope, :level, :message, :context::jsonb)
                    """,
                    {
                        "scope": "ingest_ticks",
                        "level": "error",
                        "message": "timestamp_invalid",
                        "context": json.dumps({"market_id": t.market_id, "venue_id": t.venue_id, "error": err_ts}),
                    },
                )
                continue

            await upsert_tick(session, t)

        await session.commit()
        log.info("ingest_ticks_done", n=len(ticks), mock_mode=settings.mock_mode)


if __name__ == "__main__":
    asyncio.run(run())

