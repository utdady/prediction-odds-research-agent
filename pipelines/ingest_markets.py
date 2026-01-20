from __future__ import annotations

import asyncio

from pm_agent.config import settings
from pm_agent.connectors.mock import MockConnector
from pm_agent.db import get_session
from pm_agent.logging import configure_logging
import structlog

from pm_agent.repo.upserts import upsert_market, upsert_venue
from pm_agent.repo.upserts import upsert_event


log = structlog.get_logger(__name__)


async def run() -> None:
    configure_logging()
    async with get_session() as session:
        kalshi = MockConnector("kalshi", "data/mock/kalshi_markets.json", "data/mock/kalshi_ticks.json")
        poly = MockConnector("polymarket", "data/mock/poly_markets.json", "data/mock/poly_ticks.json")

        await upsert_venue(session, "kalshi", "Kalshi")
        await upsert_venue(session, "polymarket", "Polymarket")

        markets = (await kalshi.fetch_markets()) + (await poly.fetch_markets())
        for m in markets:
            # Ensure FK target exists (markets.event_id -> events.event_id)
            if m.event_id:
                await upsert_event(
                    session,
                    event_id=m.event_id,
                    family=settings.feature_family,
                    title=m.title,
                    resolution_ts=m.resolution_ts,
                )
            await upsert_market(session, m)

        await session.commit()
        log.info("ingest_markets_done", n=len(markets), mock_mode=settings.mock_mode)


if __name__ == "__main__":
    asyncio.run(run())

