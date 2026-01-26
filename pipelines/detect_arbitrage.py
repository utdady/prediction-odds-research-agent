"""Detect arbitrage opportunities across venues."""
from __future__ import annotations

import asyncio

from pm_agent.arbitrage.detector import ArbitrageDetector
from pm_agent.db import get_session
from pm_agent.logging import configure_logging
from pm_agent.sql import fetch_all

import structlog


log = structlog.get_logger(__name__)


async def run() -> None:
    configure_logging()
    detector = ArbitrageDetector(min_spread=0.03)

    async with get_session() as session:
        # Fetch markets with probabilities (from latest ticks)
        rows = await fetch_all(
            session,
            """
            SELECT DISTINCT ON (m.market_id, m.venue_id)
                m.market_id, 
                m.event_id, 
                m.venue_id, 
                t.p_norm as probability, 
                m.title
            FROM markets m
            INNER JOIN odds_ticks t ON m.market_id = t.market_id
            WHERE t.p_norm IS NOT NULL
            ORDER BY m.market_id, m.venue_id, t.tick_ts DESC
            """,
        )

        if not rows:
            log.info("arbitrage_no_markets")
            return

        # Convert to list of dicts
        markets = [
            {
                "event_id": r["event_id"],
                "venue_id": r["venue_id"],
                "probability": float(r["probability"]) if r["probability"] else None,
            }
            for r in rows
        ]

        opportunities = detector.find_arbitrage_opportunities(markets)

        if opportunities:
            log.info(
                "arbitrage_opportunities_found",
                n=len(opportunities),
                opportunities=[
                    {
                        "event_id": opp.event_id,
                        "spread": opp.spread,
                        "expected_profit": opp.expected_profit,
                        "action": opp.action,
                    }
                    for opp in opportunities
                ],
            )
        else:
            log.info("arbitrage_no_opportunities")

        # TODO: Store opportunities in database or send alerts


if __name__ == "__main__":
    asyncio.run(run())

