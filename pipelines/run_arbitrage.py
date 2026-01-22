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
    """Detect arbitrage opportunities."""
    configure_logging()
    
    detector = ArbitrageDetector(min_spread=0.03)
    
    async with get_session() as session:
        # Fetch markets from all venues
        rows = await fetch_all(
            session,
            """
            SELECT m.market_id, m.event_id, m.venue_id, m.probability, m.title
            FROM markets m
            WHERE m.status = 'open'
            ORDER BY m.event_id, m.venue_id
            """,
        )
        
        if not rows:
            log.info("arbitrage_no_markets")
            return
        
        # Convert to market dicts
        markets = [
            {
                "event_id": row["event_id"],
                "venue_id": row["venue_id"],
                "probability": float(row["probability"]) if row["probability"] else None,
            }
            for row in rows
            if row["probability"] is not None
        ]
        
        opportunities = detector.find_arbitrage_opportunities(markets)
        
        if opportunities:
            log.info(
                "arbitrage_opportunities_found",
                n=len(opportunities),
                top_5=[{
                    "event_id": opp.event_id,
                    "spread": opp.spread,
                    "expected_profit": opp.expected_profit,
                } for opp in opportunities[:5]],
            )
        else:
            log.info("arbitrage_no_opportunities")


if __name__ == "__main__":
    asyncio.run(run())

