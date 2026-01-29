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
        # Use live connectors if not in mock mode and credentials are provided
        if settings.mock_mode:
            kalshi = MockConnector("kalshi", "data/mock/kalshi_markets.json", "data/mock/kalshi_ticks.json")
            poly = MockConnector("polymarket", "data/mock/polymarket_markets.json", "data/mock/polymarket_ticks.json")
        else:
            from pm_agent.connectors.kalshi import KalshiConnector
            from pm_agent.connectors.polymarket import PolymarketConnector
            
            kalshi = KalshiConnector(
                api_key=settings.kalshi_api_key,
                api_secret=settings.kalshi_api_secret,
            )
            poly = PolymarketConnector(api_key=settings.polymarket_api_key)

        await upsert_venue(session, "kalshi", "Kalshi")
        await upsert_venue(session, "polymarket", "Polymarket")

        try:
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
        finally:
            # Close live connectors if they have a close method
            if hasattr(kalshi, "close"):
                await kalshi.close()
            if hasattr(poly, "close"):
                await poly.close()


if __name__ == "__main__":
    asyncio.run(run())

