"""Comprehensive Kalshi API tests."""
import asyncio
import os
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pm_agent.connectors.kalshi import KalshiConnector
from pm_agent.config import settings


@pytest.mark.asyncio
async def test_kalshi_public_markets():
    """Test fetching markets from public endpoint."""
    connector = KalshiConnector()  # No credentials
    
    try:
        markets = await connector.fetch_markets(limit=10)
        
        assert isinstance(markets, list), "Should return a list"
        if markets:
            assert hasattr(markets[0], "market_id"), "Market should have market_id"
            assert hasattr(markets[0], "title"), "Market should have title"
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_kalshi_authenticated_markets():
    """Test fetching markets with authentication."""
    if not settings.kalshi_api_key or not settings.kalshi_api_secret:
        pytest.skip("Kalshi credentials not configured")
    
    connector = KalshiConnector(
        api_key=settings.kalshi_api_key,
        api_secret=settings.kalshi_api_secret
    )
    
    try:
        markets = await connector.fetch_markets(limit=10)
        
        assert isinstance(markets, list), "Should return a list"
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_kalshi_orderbook():
    """Test fetching orderbook data."""
    connector = KalshiConnector()
    
    try:
        # First get some markets
        markets = await connector.fetch_markets(limit=3)
        
        if not markets:
            pytest.skip("No markets available to test orderbook")
        
        market_ids = [m.market_id for m in markets[:2]]
        ticks = await connector.fetch_ticks(market_ids)
        
        assert isinstance(ticks, list), "Should return a list"
        if ticks:
            assert hasattr(ticks[0], "market_id"), "Tick should have market_id"
            assert hasattr(ticks[0], "p_norm"), "Tick should have p_norm"
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_kalshi_rate_limiting():
    """Test that rate limiting works."""
    connector = KalshiConnector()
    
    try:
        # Make multiple requests quickly
        tasks = [connector.fetch_markets(limit=1) for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should not all fail due to rate limiting
        successes = sum(1 for r in results if not isinstance(r, Exception))
        assert successes > 0, "At least some requests should succeed"
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_kalshi_error_handling():
    """Test error handling for invalid requests."""
    connector = KalshiConnector()
    
    try:
        # Try to fetch orderbook for non-existent market
        ticks = await connector.fetch_ticks(["INVALID_MARKET_ID"])
        
        # Should handle gracefully (return empty list or log warning)
        assert isinstance(ticks, list), "Should return a list even on error"
    finally:
        await connector.close()

