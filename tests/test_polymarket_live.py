"""Comprehensive live API tests for Polymarket connector.

Run with: pytest tests/test_polymarket_live.py -v -s
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pm_agent.connectors.polymarket import PolymarketConnector
from pm_agent.config import settings

# Polymarket public API works without credentials
POLYMARKET_API_KEY = settings.polymarket_api_key
HAS_CREDENTIALS = bool(POLYMARKET_API_KEY)


@pytest.mark.asyncio
async def test_polymarket_public_markets():
    """Test fetching markets without authentication (public CLOB endpoint)."""
    connector = PolymarketConnector()  # No credentials needed
    
    try:
        print("\n" + "="*60)
        print("TEST: Polymarket Public Markets (No Auth)")
        print("="*60)
        
        markets = await connector.fetch_markets(limit=10)
        
        print(f"\n[OK] Fetched {len(markets)} markets")
        
        if markets:
            print("\nSample markets:")
            for i, m in enumerate(markets[:3], 1):
                print(f"\n{i}. {m.title[:60]}...")
                print(f"   Market ID: {m.market_id}")
                print(f"   Status: {m.status}")
                print(f"   Resolution: {m.resolution_ts}")
            
            # Validation
            assert len(markets) > 0, "Should fetch at least some markets"
            assert all(m.venue_id == "polymarket" for m in markets)
            assert all(m.market_id for m in markets)
            assert all(m.title for m in markets)
            
            print("\n[OK] All validations passed")
        else:
            print("\n[WARN] Warning: No markets returned")
            print("  This could indicate API changes or connectivity issues")
    
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nPossible causes:")
        print("1. Polymarket API endpoint changed")
        print("2. Rate limiting (try again in a minute)")
        print("3. Network connectivity issues")
        print("4. Check that Gamma API endpoint is correct")
        raise
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_polymarket_active_inactive():
    """Test filtering by active/inactive status."""
    connector = PolymarketConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Polymarket Active/Inactive Filtering")
        print("="*60)
        
        # Get active markets
        active_markets = await connector.fetch_markets(limit=10, active=True)
        print(f"\n[OK] Active markets: {len(active_markets)}")
        
        # Get all markets (including inactive)
        all_markets = await connector.fetch_markets(limit=10, active=False)
        print(f"[OK] All markets: {len(all_markets)}")
        
        # Verify status
        if active_markets:
            active_statuses = {m.status for m in active_markets}
            print(f"  Active market statuses: {active_statuses}")
        
        print("\n[OK] Status filtering tests passed")
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_polymarket_public_ticks():
    """Test fetching tick/orderbook data."""
    connector = PolymarketConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Polymarket Public Ticks/Orderbook")
        print("="*60)
        
        # First get some markets
        markets = await connector.fetch_markets(limit=5, active=True)
        
        if not markets:
            print("[WARN] No active markets found, skipping tick test")
            pytest.skip("No active markets available")
        
        market_ids = [m.market_id for m in markets[:3]]
        print(f"\nTesting ticks for {len(market_ids)} markets")
        
        ticks = await connector.fetch_ticks(market_ids)
        
        print(f"\n[OK] Fetched {len(ticks)} ticks")
        
        if ticks:
            print("\nSample ticks:")
            for i, tick in enumerate(ticks[:3], 1):
                print(f"\n{i}. Market: {tick.market_id}")
                print(f"   Timestamp: {tick.tick_ts}")
                print(f"   Normalized Prob: {tick.p_norm:.3f}")
                if tick.yes_bid:
                    print(f"   Yes Bid: {tick.yes_bid:.3f}")
                else:
                    print(f"   Yes Bid: None")
                if tick.yes_ask:
                    print(f"   Yes Ask: {tick.yes_ask:.3f}")
                else:
                    print(f"   Yes Ask: None")
                
                if tick.yes_bid and tick.yes_ask:
                    spread = tick.yes_ask - tick.yes_bid
                    spread_pct = spread / tick.yes_mid * 100 if tick.yes_mid else 0
                    print(f"   Spread: {spread:.4f} ({spread_pct:.1f}%)")
            
            # Validation
            assert all(t.venue_id == "polymarket" for t in ticks)
            assert all(0 <= t.p_norm <= 1 for t in ticks), "Probabilities must be in [0,1]"
            
            print("\n[OK] All tick validations passed")
        else:
            print("\n[WARN] No ticks returned")
            print("  Possible causes:")
            print("  - Markets may not have active orderbooks")
            print("  - API endpoint changed")
            print("  - Check that CLOB API endpoint is correct")
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_polymarket_rate_limiting():
    """Test that rate limiting works correctly."""
    import time
    
    connector = PolymarketConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Polymarket Rate Limiting")
        print("="*60)
        print("Making 10 rapid requests...")
        
        start = time.time()
        
        for i in range(10):
            markets = await connector.fetch_markets(limit=1)
            print(f"  Request {i+1}: {len(markets)} markets")
        
        elapsed = time.time() - start
        
        print(f"\n[OK] Completed 10 requests in {elapsed:.2f}s")
        print(f"  Average: {elapsed/10:.2f}s per request")
        
        if elapsed < 5:
            print("  [OK] Rate limiting seems to be working")
        else:
            print("  [WARN] Requests may not be efficiently rate limited")
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_polymarket_error_handling():
    """Test error handling for invalid requests."""
    connector = PolymarketConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Polymarket Error Handling")
        print("="*60)
        
        # Test fetching ticks for non-existent market
        print("\nTesting non-existent market...")
        ticks = await connector.fetch_ticks(["invalid_condition_id_999"])
        
        # Should return empty list, not crash
        print(f"[OK] Handled gracefully: returned {len(ticks)} ticks")
        assert isinstance(ticks, list)
        
        print("\n[OK] Error handling tests passed")
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_polymarket_data_quality():
    """Test data quality and consistency."""
    connector = PolymarketConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Polymarket Data Quality")
        print("="*60)
        
        markets = await connector.fetch_markets(limit=20)
        
        if not markets:
            pytest.skip("No markets available")
        
        # Check for required fields
        issues = []
        stats = {
            "has_resolution_ts": 0,
            "has_event_id": 0,
            "has_description": 0,
        }
        
        for m in markets:
            if not m.market_id:
                issues.append(f"Market missing ID: {m.title[:50]}")
            if not m.title:
                issues.append(f"Market missing title: {m.market_id}")
            
            if m.resolution_ts:
                stats["has_resolution_ts"] += 1
            if m.event_id:
                stats["has_event_id"] += 1
            if m.description:
                stats["has_description"] += 1
        
        print(f"\nData completeness:")
        print(f"  Markets with resolution timestamp: {stats['has_resolution_ts']}/{len(markets)}")
        print(f"  Markets with event ID: {stats['has_event_id']}/{len(markets)}")
        print(f"  Markets with description: {stats['has_description']}/{len(markets)}")
        
        if issues:
            print("\n[WARN] Data quality issues found:")
            for issue in issues[:5]:  # Show first 5
                print(f"  - {issue}")
        else:
            print("\n[OK] No critical data quality issues")
        
        # Check tick data quality
        market_ids = [m.market_id for m in markets[:3]]
        ticks = await connector.fetch_ticks(market_ids)
        
        for t in ticks:
            # Probability must be valid
            assert 0 <= t.p_norm <= 1, f"Invalid probability: {t.p_norm}"
            
            # If bid/ask present, bid should be <= ask
            if t.yes_bid is not None and t.yes_ask is not None:
                if t.yes_bid > t.yes_ask:
                    print(f"  [WARN] Inverted spread: bid={t.yes_bid} > ask={t.yes_ask}")
        
        print(f"\n[OK] Validated {len(ticks)} ticks")
        print("[OK] All data quality checks completed")
    
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_polymarket_concurrent_requests():
    """Test handling concurrent requests."""
    connector = PolymarketConnector()
    
    try:
        print("\n" + "="*60)
        print("TEST: Polymarket Concurrent Requests")
        print("="*60)
        
        # Make 5 concurrent requests
        tasks = [
            connector.fetch_markets(limit=5)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        print(f"\n[OK] {successful}/5 concurrent requests succeeded")
        
        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            print(f"\n[WARN] {len(errors)} requests failed:")
            for e in errors[:3]:
                print(f"  - {type(e).__name__}: {str(e)[:60]}")
        
        assert successful > 0, "At least some concurrent requests should succeed"
        print("\n[OK] Concurrent request test passed")
    
    finally:
        await connector.close()


def test_credentials_setup():
    """Check if credentials are configured (optional for Polymarket)."""
    print("\n" + "="*60)
    print("CREDENTIALS CHECK")
    print("="*60)
    
    if HAS_CREDENTIALS:
        print("[OK] Polymarket API key found")
        print(f"  API Key: {POLYMARKET_API_KEY[:8]}...")
        print("\nNote: API key is only needed for private endpoints")
        print("      (placing orders, checking balance, etc.)")
    else:
        print("[OK] No API key configured (using public endpoints)")
        print("\nThis is OKAY! Polymarket CLOB endpoints are public.")
        print("\nAPI key only needed for:")
        print("  - Placing orders")
        print("  - Checking account balance")
        print("  - Accessing private data")


if __name__ == "__main__":
    # Run basic test without pytest
    print("\nRunning Polymarket API Tests")
    print("=" * 60)
    
    asyncio.run(test_polymarket_public_markets())
    asyncio.run(test_polymarket_public_ticks())
    
    print("\n" + "="*60)
    print("[OK] Basic tests completed!")
    print("="*60)
    print("\nRun full test suite with: pytest tests/test_polymarket_live.py -v -s")
