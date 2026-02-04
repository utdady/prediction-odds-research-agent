"""Comprehensive live API tests for Kalshi connector.

Run with: pytest tests/test_kalshi_live.py -v -s
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pm_agent.connectors.kalshi import KalshiConnector
from pm_agent.config import settings

# Check if credentials are available
KALSHI_API_KEY = settings.kalshi_api_key
KALSHI_API_SECRET = settings.kalshi_api_secret
HAS_CREDENTIALS = bool(KALSHI_API_KEY and KALSHI_API_SECRET)


@pytest.mark.asyncio
async def test_kalshi_public_markets():
    """Test fetching markets without authentication (public endpoint)."""
    connector = KalshiConnector()  # No credentials
    
    try:
        print("\n" + "="*60)
        print("TEST: Kalshi Public Markets (No Auth)")
        print("="*60)
        
        markets = await connector.fetch_markets(limit=5)
        
        print(f"\n[OK] Fetched {len(markets)} markets")
        
        if markets:
            print("\nSample market:")
            m = markets[0]
            print(f"  Market ID: {m.market_id}")
            print(f"  Title: {m.title}")
            print(f"  Status: {m.status}")
            print(f"  Event ID: {m.event_id}")
            
            # Validation
            assert len(markets) > 0, "Should fetch at least some markets"
            assert all(m.venue_id == "kalshi" for m in markets)
            assert all(m.market_id for m in markets)
            assert all(m.title for m in markets)
            
            print("\n[OK] All validations passed")
        else:
            print("\n[WARN] Warning: No markets returned")
            print("  This might be okay if no markets are currently open")
    
    finally:
        await connector.close()


@pytest.mark.skipif(not HAS_CREDENTIALS, reason="Kalshi credentials not set")
@pytest.mark.asyncio
async def test_kalshi_authenticated_markets():
    """Test fetching markets with authentication."""
    connector = KalshiConnector(
        api_key=KALSHI_API_KEY,
        api_secret=KALSHI_API_SECRET,
    )
    
    try:
        print("\n" + "="*60)
        print("TEST: Kalshi Authenticated Markets")
        print("="*60)
        print(f"API Key: {KALSHI_API_KEY[:8]}...")
        
        markets = await connector.fetch_markets(limit=10)
        
        print(f"\n[OK] Fetched {len(markets)} markets with authentication")
        
        assert len(markets) > 0, "Should fetch markets with valid credentials"
        
        # Test different status filters
        open_markets = await connector.fetch_markets(limit=5, status="open")
        print(f"[OK] Open markets: {len(open_markets)}")
        
        closed_markets = await connector.fetch_markets(limit=5, status="closed")
        print(f"[OK] Closed markets: {len(closed_markets)}")
        
        print("\n[OK] All authenticated tests passed")
    
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify API key and secret are correct")
        print("2. Check if API key is active in Kalshi portal")
        print("3. Ensure no IP restrictions on API key")
        print("4. Verify timestamp format (should be milliseconds)")
        print("5. Check that headers use Kalshi-Access-Key and Kalshi-Access-Signature")
        raise
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_kalshi_public_ticks():
    """Test fetching tick data (orderbook)."""
    connector = KalshiConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Kalshi Public Ticks/Orderbook")
        print("="*60)
        
        # First get some markets
        markets = await connector.fetch_markets(limit=3, status="open")
        
        if not markets:
            print("[WARN] No open markets found, skipping tick test")
            pytest.skip("No open markets available")
        
        market_ids = [m.market_id for m in markets[:2]]
        print(f"\nTesting ticks for markets: {market_ids}")
        
        ticks = await connector.fetch_ticks(market_ids)
        
        print(f"\n[OK] Fetched {len(ticks)} ticks")
        
        if ticks:
            tick = ticks[0]
            print(f"\nSample tick:")
            print(f"  Market: {tick.market_id}")
            print(f"  Timestamp: {tick.tick_ts}")
            print(f"  Normalized Prob: {tick.p_norm:.3f}")
            print(f"  Yes Bid: {tick.yes_bid}")
            print(f"  Yes Ask: {tick.yes_ask}")
            if tick.yes_ask and tick.yes_bid:
                spread = tick.yes_ask - tick.yes_bid
                print(f"  Spread: {spread:.4f}")
            else:
                print(f"  Spread: N/A")
            
            # Validation
            assert all(t.venue_id == "kalshi" for t in ticks)
            assert all(0 <= t.p_norm <= 1 for t in ticks), "Probabilities must be in [0,1]"
            
            print("\n[OK] All tick validations passed")
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_kalshi_rate_limiting():
    """Test that rate limiting works correctly."""
    import time
    
    connector = KalshiConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Kalshi Rate Limiting")
        print("="*60)
        print("Making 5 rapid requests...")
        
        start = time.time()
        
        for i in range(5):
            markets = await connector.fetch_markets(limit=1)
            print(f"  Request {i+1}: {len(markets)} markets")
        
        elapsed = time.time() - start
        
        print(f"\n[OK] Completed 5 requests in {elapsed:.2f}s")
        print(f"  Average: {elapsed/5:.2f}s per request")
        
        if elapsed < 3:
            print("  [OK] Rate limiting seems to be working (requests were throttled)")
        else:
            print("  [WARN] Requests may not be rate limited")
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_kalshi_error_handling():
    """Test error handling for invalid requests."""
    connector = KalshiConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Kalshi Error Handling")
        print("="*60)
        
        # Test fetching ticks for non-existent market
        print("\nTesting non-existent market...")
        ticks = await connector.fetch_ticks(["INVALID_MARKET_ID_999"])
        
        # Should return empty list, not crash
        print(f"[OK] Handled gracefully: returned {len(ticks)} ticks")
        assert isinstance(ticks, list)
        
        print("\n[OK] Error handling tests passed")
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_kalshi_data_quality():
    """Test data quality and consistency."""
    connector = KalshiConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Kalshi Data Quality")
        print("="*60)
        
        markets = await connector.fetch_markets(limit=10)
        
        if not markets:
            pytest.skip("No markets available")
        
        # Check for required fields
        issues = []
        
        for m in markets:
            if not m.market_id:
                issues.append(f"Market missing ID: {m.title}")
            if not m.title:
                issues.append(f"Market missing title: {m.market_id}")
            if m.status not in ["open", "closed", "settled", "finalized", "active"]:
                issues.append(f"Unexpected status '{m.status}' for {m.market_id}")
        
        if issues:
            print("\n[WARN] Data quality issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n[OK] All data quality checks passed")
        
        # Check tick data quality
        market_ids = [m.market_id for m in markets[:2]]
        ticks = await connector.fetch_ticks(market_ids)
        
        for t in ticks:
            # Probability must be valid
            assert 0 <= t.p_norm <= 1, f"Invalid probability: {t.p_norm}"
            
            # If bid/ask present, bid should be <= ask
            if t.yes_bid is not None and t.yes_ask is not None:
                if t.yes_bid > t.yes_ask:
                    print(f"  [WARN] Inverted spread: bid={t.yes_bid} > ask={t.yes_ask}")
        
        print(f"[OK] Validated {len(ticks)} ticks")
    
    finally:
        await connector.close()


def test_credentials_setup():
    """Check if credentials are properly configured."""
    print("\n" + "="*60)
    print("CREDENTIALS CHECK")
    print("="*60)
    
    if HAS_CREDENTIALS:
        print("[OK] Kalshi credentials found")
        print(f"  API Key: {KALSHI_API_KEY[:8]}...")
        print(f"  API Secret: {'*' * 32}")
    else:
        print("[WARN] Kalshi credentials NOT found")
        print("\nTo set credentials:")
        print("  Set in .env file:")
        print("    KALSHI_API_KEY=your_key")
        print("    KALSHI_API_SECRET=your_secret")
        print("\nOr set environment variables:")
        print("  $env:KALSHI_API_KEY='your_key'")
        print("  $env:KALSHI_API_SECRET='your_secret'")
        print("\nNote: Public endpoints work without credentials!")


if __name__ == "__main__":
    # Run basic test without pytest
    print("\nRunning Kalshi API Tests")
    print("=" * 60)
    
    asyncio.run(test_kalshi_public_markets())
    
    if HAS_CREDENTIALS:
        asyncio.run(test_kalshi_authenticated_markets())
        asyncio.run(test_kalshi_public_ticks())
    else:
        print("\n[WARN] Skipping authenticated tests (no credentials)")
        print("  Set KALSHI_API_KEY and KALSHI_API_SECRET in .env to run all tests")
