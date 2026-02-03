"""Kalshi API connector for live prediction market data."""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
from datetime import datetime
from typing import Any

import httpx
import structlog

from pm_agent.connectors.rate_limit import KALSHI_RATE_LIMITER, retry_with_backoff
from pm_agent.schemas import NormalizedMarket, NormalizedTick

log = structlog.get_logger(__name__)


class KalshiConnector:
    """Connector for Kalshi prediction market API."""

    BASE_URL = "https://api.cx.kalshi.com/trade-api/v2"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        """
        Initialize Kalshi connector.
        
        Args:
            api_key: Optional Kalshi API key (for authenticated endpoints).
                     If None, uses public market data endpoints (no auth required).
            api_secret: Optional Kalshi API secret (required if api_key provided)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_auth = bool(api_key and api_secret)
        self.client = httpx.AsyncClient(timeout=30.0)
        
        if not self.use_auth:
            log.info("kalshi_public_mode", message="Using public endpoints (no authentication)")

    def _generate_auth_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        """Generate authentication headers for Kalshi API."""
        if not self.api_key or not self.api_secret:
            return {}

        timestamp = str(int(time.time()))
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        auth_string = base64.b64encode(f"{self.api_key}:{signature}".encode()).decode()

        return {
            "Authorization": f"Basic {auth_string}",
            "Kalshi-Access-Timestamp": timestamp,
        }

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    async def fetch_markets(
        self,
        limit: int = 100,
        status: str = "open",
        event_ticker: str | None = None,
    ) -> list[NormalizedMarket]:
        """
        Fetch active markets from Kalshi.
        
        Args:
            limit: Maximum number of markets to fetch
            status: Market status filter (open, closed, etc.)
            event_ticker: Optional event ticker filter
        """
        await KALSHI_RATE_LIMITER.acquire()
        try:
            path = "/markets"
            params: dict[str, Any] = {
                "limit": limit,
                "status": status,
            }
            if event_ticker:
                params["event_ticker"] = event_ticker

            headers = self._generate_auth_headers("GET", path)
            response = await self.client.get(
                f"{self.BASE_URL}{path}",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            markets: list[NormalizedMarket] = []
            for m in data.get("markets", []):
                resolution_ts = None
                if m.get("expiration_ts"):
                    resolution_ts = datetime.fromtimestamp(m["expiration_ts"])

                markets.append(
                    NormalizedMarket(
                        venue_id="kalshi",
                        market_id=m["ticker"],
                        event_id=m.get("event_ticker"),
                        title=m.get("title", ""),
                        description=m.get("subtitle", ""),
                        status=m.get("status", "active"),
                        resolution_ts=resolution_ts,
                        raw=m,
                    )
                )

            log.info("kalshi_markets_fetched", n=len(markets))
            return markets

        except Exception as e:
            log.error("kalshi_fetch_markets_error", error=str(e))
            return []

    async def fetch_ticks(self, market_ids: list[str]) -> list[NormalizedTick]:
        """
        Fetch recent ticks (order book) for given markets.
        
        Args:
            market_ids: List of market tickers to fetch
        """
        all_ticks: list[NormalizedTick] = []

        for market_id in market_ids:
            await KALSHI_RATE_LIMITER.acquire()
            try:
                path = f"/markets/{market_id}/orderbook"
                headers = self._generate_auth_headers("GET", path)

                response = await self.client.get(
                    f"{self.BASE_URL}{path}",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                # Extract best bid/ask from orderbook
                yes_side = data.get("yes", {})
                no_side = data.get("no", {})

                yes_bid = yes_side.get("bids", [{}])[0].get("price") if yes_side.get("bids") else None
                yes_ask = yes_side.get("asks", [{}])[0].get("price") if yes_side.get("asks") else None
                no_bid = no_side.get("bids", [{}])[0].get("price") if no_side.get("bids") else None
                no_ask = no_side.get("asks", [{}])[0].get("price") if no_side.get("asks") else None

                # Calculate mid prices
                yes_mid = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else None
                no_mid = (no_bid + no_ask) / 2 if (no_bid and no_ask) else None

                # Normalized probability (yes_mid if available, else 1 - no_mid)
                p_norm = yes_mid if yes_mid else (1 - no_mid) if no_mid else 0.5

                all_ticks.append(
                    NormalizedTick(
                        venue_id="kalshi",
                        market_id=market_id,
                        tick_ts=datetime.utcnow(),
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        no_bid=no_bid,
                        no_ask=no_ask,
                        yes_mid=yes_mid,
                        no_mid=no_mid,
                        p_norm=p_norm,
                        volume=None,  # Kalshi orderbook doesn't provide volume
                        raw=data,
                    )
                )

            except Exception as e:
                log.warning("kalshi_fetch_tick_error", market_id=market_id, error=str(e))
                continue

        log.info("kalshi_ticks_fetched", n=len(all_ticks))
        return all_ticks

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

