"""Polymarket API connector for live prediction market data."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import structlog

from pm_agent.connectors.rate_limit import POLYMARKET_RATE_LIMITER, retry_with_backoff
from pm_agent.schemas import NormalizedMarket, NormalizedTick

log = structlog.get_logger(__name__)


class PolymarketConnector:
    """Connector for Polymarket prediction market API (GraphQL)."""

    GRAPHQL_URL = "https://clob.polymarket.com/graphql"

    def __init__(self, api_key: str | None = None):
        """
        Initialize Polymarket connector.
        
        Args:
            api_key: Optional API key. Public GraphQL endpoints (CLOB/orderbook) work without auth.
                     Auth only needed for private endpoints (trading, account info).
        """
        self.api_key = api_key
        self.use_auth = bool(api_key)
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)
        
        if not self.use_auth:
            log.info("polymarket_public_mode", message="Using public GraphQL endpoints (no authentication)")

    async def _graphql_query(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a GraphQL query."""
        response = await self.client.post(
            self.GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
        )
        response.raise_for_status()
        return response.json()

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    async def fetch_markets(
        self,
        limit: int = 100,
        active: bool = True,
    ) -> list[NormalizedMarket]:
        """
        Fetch active markets from Polymarket.
        
        Args:
            limit: Maximum number of markets to fetch
            active: Only fetch active markets
        """
        await POLYMARKET_RATE_LIMITER.acquire()
        try:
            query = """
            query GetMarkets($limit: Int, $active: Boolean) {
                markets(limit: $limit, active: $active) {
                    id
                    question
                    slug
                    description
                    endDate
                    active
                    conditionId
                    outcomes {
                        id
                        title
                    }
                }
            }
            """

            data = await self._graphql_query(query, {"limit": limit, "active": active})
            markets_data = data.get("data", {}).get("markets", [])

            markets: list[NormalizedMarket] = []
            for m in markets_data:
                resolution_ts = None
                if m.get("endDate"):
                    resolution_ts = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))

                markets.append(
                    NormalizedMarket(
                        venue_id="polymarket",
                        market_id=m["id"],
                        event_id=m.get("conditionId"),
                        title=m.get("question", ""),
                        description=m.get("description", ""),
                        status="active" if m.get("active") else "closed",
                        resolution_ts=resolution_ts,
                        raw=m,
                    )
                )

            log.info("polymarket_markets_fetched", n=len(markets))
            return markets

        except Exception as e:
            log.error("polymarket_fetch_markets_error", error=str(e))
            return []

    async def fetch_ticks(self, market_ids: list[str]) -> list[NormalizedTick]:
        """
        Fetch current prices for given markets.
        
        Args:
            market_ids: List of market IDs (condition IDs) to fetch
        """
        all_ticks: list[NormalizedTick] = []

        for market_id in market_ids:
            await POLYMARKET_RATE_LIMITER.acquire()
            try:
                # Fetch orderbook for this market
                query = """
                query GetOrderbook($conditionId: String!) {
                    orderbook(conditionId: $conditionId) {
                        bids {
                            price
                            size
                        }
                        asks {
                            price
                            size
                        }
                    }
                }
                """

                data = await self._graphql_query(query, {"conditionId": market_id})
                orderbook = data.get("data", {}).get("orderbook", {})

                bids = orderbook.get("bids", [])
                asks = orderbook.get("asks", [])

                # Best bid/ask (highest bid, lowest ask)
                yes_bid = float(bids[0]["price"]) if bids else None
                yes_ask = float(asks[0]["price"]) if asks else None

                # Polymarket uses probability directly (0-1)
                p_norm = yes_bid if yes_bid else (yes_ask if yes_ask else 0.5)

                # Calculate mid
                yes_mid = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else p_norm
                no_mid = 1 - yes_mid if yes_mid else None

                all_ticks.append(
                    NormalizedTick(
                        venue_id="polymarket",
                        market_id=market_id,
                        tick_ts=datetime.utcnow(),
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        no_bid=1 - yes_ask if yes_ask else None,
                        no_ask=1 - yes_bid if yes_bid else None,
                        yes_mid=yes_mid,
                        no_mid=no_mid,
                        p_norm=p_norm,
                        volume=None,  # Polymarket orderbook doesn't always provide volume
                        raw=orderbook,
                    )
                )

            except Exception as e:
                log.warning("polymarket_fetch_tick_error", market_id=market_id, error=str(e))
                continue

        log.info("polymarket_ticks_fetched", n=len(all_ticks))
        return all_ticks

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

