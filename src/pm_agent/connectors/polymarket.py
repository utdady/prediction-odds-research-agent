"""Fixed Polymarket API connector with current CLOB API structure."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import structlog

from pm_agent.connectors.rate_limit import POLYMARKET_RATE_LIMITER, retry_with_backoff
from pm_agent.schemas import NormalizedMarket, NormalizedTick

log = structlog.get_logger(__name__)


class PolymarketConnector:
    """Connector for Polymarket CLOB (Central Limit Order Book) API.
    
    API Documentation: https://docs.polymarket.com/
    CLOB API: https://clob.polymarket.com/
    
    Note: Most CLOB endpoints are public and don't require authentication.
    """

    # Polymarket uses multiple endpoints
    CLOB_API_URL = "https://clob.polymarket.com"
    GAMMA_API_URL = "https://gamma-api.polymarket.com"

    def __init__(self, api_key: str | None = None):
        """
        Initialize Polymarket connector.
        
        Args:
            api_key: Optional API key (not needed for public market data)
        
        Note:
            Public endpoints (no auth required):
            - /markets - Get all markets
            - /book - Get order book for a market
            - /prices - Get current prices
            
            Private endpoints (require auth):
            - /order - Place orders
            - /balance - Check balance
        """
        self.api_key = api_key
        self.use_auth = bool(api_key)
        
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)
        
        if not self.use_auth:
            log.info("polymarket_public_mode", message="Using public CLOB endpoints (no authentication)")
        else:
            log.info("polymarket_auth_mode", message="Using authenticated endpoints")

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    async def fetch_markets(
        self,
        limit: int = 100,
        active: bool = True,
    ) -> list[NormalizedMarket]:
        """
        Fetch markets from Polymarket.
        
        Public endpoint - works without authentication.
        Uses Gamma API which provides market metadata.
        
        Args:
            limit: Maximum number of markets
            active: Only fetch active markets
        
        Returns:
            List of normalized markets
        """
        await POLYMARKET_RATE_LIMITER.acquire()
        
        try:
            # Use Gamma API for market data
            params: dict[str, Any] = {
                "limit": limit,
            }
            
            if active:
                params["active"] = "true"
            
            response = await self.client.get(
                f"{self.GAMMA_API_URL}/markets",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            
            markets: list[NormalizedMarket] = []
            
            # Gamma API returns array of markets directly
            markets_data = data if isinstance(data, list) else data.get("data", [])
            
            for m in markets_data:
                # Parse end date
                resolution_ts = None
                if m.get("end_date_iso"):
                    try:
                        resolution_ts = datetime.fromisoformat(
                            m["end_date_iso"].replace("Z", "+00:00")
                        )
                    except Exception as e:
                        log.warning("polymarket_parse_time_failed", error=str(e))
                
                # Extract condition ID (used for fetching prices)
                condition_id = m.get("condition_id") or m.get("id")
                
                markets.append(
                    NormalizedMarket(
                        venue_id="polymarket",
                        market_id=condition_id,
                        event_id=m.get("group_item_title"),  # Event grouping
                        title=m.get("question", ""),
                        description=m.get("description", ""),
                        status="active" if m.get("active", True) else "closed",
                        resolution_ts=resolution_ts,
                        raw=m,
                    )
                )
            
            log.info("polymarket_markets_fetched", n=len(markets), active=active)
            return markets
        
        except httpx.HTTPStatusError as e:
            log.error(
                "polymarket_http_error",
                status=e.response.status_code,
                body=e.response.text[:500]
            )
            return []
        except Exception as e:
            log.error("polymarket_fetch_markets_error", error=str(e))
            return []

    async def fetch_ticks(self, market_ids: list[str]) -> list[NormalizedTick]:
        """
        Fetch current prices/orderbook for markets.
        
        Uses CLOB API to get order book data.
        Public endpoint - works without authentication.
        
        Args:
            market_ids: List of condition IDs (token IDs)
        
        Returns:
            List of normalized ticks
        """
        all_ticks: list[NormalizedTick] = []
        
        for market_id in market_ids:
            await POLYMARKET_RATE_LIMITER.acquire()
            
            try:
                # Get order book for this market
                # Polymarket CLOB uses token_id parameter
                response = await self.client.get(
                    f"{self.CLOB_API_URL}/book",
                    params={"token_id": market_id},
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract bids and asks
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                
                # Get best bid/ask (highest bid, lowest ask)
                yes_bid = None
                yes_ask = None
                
                if bids:
                    # Bids are sorted descending by price
                    best_bid = bids[0]
                    yes_bid = float(best_bid.get("price", 0))
                
                if asks:
                    # Asks are sorted ascending by price
                    best_ask = asks[0]
                    yes_ask = float(best_ask.get("price", 0))
                
                # Calculate mid price
                yes_mid = None
                if yes_bid is not None and yes_ask is not None:
                    yes_mid = (yes_bid + yes_ask) / 2
                elif yes_bid is not None:
                    yes_mid = yes_bid
                elif yes_ask is not None:
                    yes_mid = yes_ask
                
                # No side calculations
                no_bid = None
                no_ask = None
                no_mid = None
                
                if yes_ask is not None:
                    no_bid = 1 - yes_ask
                if yes_bid is not None:
                    no_ask = 1 - yes_bid
                if yes_mid is not None:
                    no_mid = 1 - yes_mid
                
                # Normalized probability
                p_norm = yes_mid if yes_mid else 0.5
                
                # Calculate volume (sum of sizes at best bid/ask)
                volume = None
                if bids or asks:
                    bid_vol = float(bids[0].get("size", 0)) if bids else 0
                    ask_vol = float(asks[0].get("size", 0)) if asks else 0
                    volume = bid_vol + ask_vol
                
                all_ticks.append(
                    NormalizedTick(
                        venue_id="polymarket",
                        market_id=market_id,
                        tick_ts=datetime.utcnow(),
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        no_bid=no_bid,
                        no_ask=no_ask,
                        yes_mid=yes_mid,
                        no_mid=no_mid,
                        p_norm=p_norm,
                        volume=volume,
                        raw=data,
                    )
                )
            
            except httpx.HTTPStatusError as e:
                log.warning(
                    "polymarket_fetch_tick_error",
                    market_id=market_id,
                    status=e.response.status_code,
                    body=e.response.text[:200]
                )
                continue
            except Exception as e:
                log.warning("polymarket_fetch_tick_error", market_id=market_id, error=str(e))
                continue
        
        log.info("polymarket_ticks_fetched", n=len(all_ticks))
        return all_ticks

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
