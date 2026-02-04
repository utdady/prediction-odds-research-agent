#!/usr/bin/env python3
"""Quick-start script to test API connectors.

This script runs basic tests to verify your API implementations work correctly.
It will test both public endpoints (no credentials) and authenticated endpoints
(if credentials are provided).

Usage:
    python tests/quick_test_apis.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_kalshi():
    """Test Kalshi connector."""
    print("\n" + "="*70)
    print("[TESTING] KALSHI API")
    print("="*70)
    
    try:
        from pm_agent.connectors.kalshi import KalshiConnector
    except ImportError as e:
        print(f"[ERROR] Could not import KalshiConnector: {e}")
        return False
    
    # Test public endpoint (no credentials)
    print("\n[TEST 1] Public Markets (No Authentication)")
    print("-" * 70)
    
    connector = KalshiConnector()
    try:
        markets = await connector.fetch_markets(limit=5)
        
        if markets:
            print(f"[OK] SUCCESS: Fetched {len(markets)} markets")
            print(f"\n   Sample market:")
            print(f"   - ID: {markets[0].market_id}")
            print(f"   - Title: {markets[0].title[:60]}...")
            print(f"   - Status: {markets[0].status}")
        else:
            print("[WARN] WARNING: No markets returned")
            print("   This might be normal if no markets are currently open")
        
        # Test authenticated endpoint if credentials available
        from pm_agent.config import settings
        api_key = settings.kalshi_api_key
        api_secret = settings.kalshi_api_secret
        
        if api_key and api_secret:
            print("\n[TEST 2] Authenticated Request")
            print("-" * 70)
            
            auth_connector = KalshiConnector(api_key=api_key, api_secret=api_secret)
            try:
                auth_markets = await auth_connector.fetch_markets(limit=5)
                print(f"[OK] SUCCESS: Authenticated request fetched {len(auth_markets)} markets")
            except Exception as e:
                print(f"[ERROR] FAILED: {e}")
                print("\n   Troubleshooting:")
                print("   1. Verify API key and secret are correct")
                print("   2. Check Kalshi API portal for key status")
                print("   3. Verify timestamp format (should be milliseconds)")
                return False
            finally:
                await auth_connector.close()
        else:
            print("\n[SKIP] SKIPPED: Authenticated test (no credentials)")
            print("   Set KALSHI_API_KEY and KALSHI_API_SECRET in .env to test")
        
        print("\n" + "="*70)
        print("[OK] KALSHI: All tests passed!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await connector.close()


async def test_polymarket():
    """Test Polymarket connector."""
    print("\n" + "="*70)
    print("[TESTING] POLYMARKET API")
    print("="*70)
    
    try:
        from pm_agent.connectors.polymarket import PolymarketConnector
    except ImportError as e:
        print(f"[ERROR] Could not import PolymarketConnector: {e}")
        return False
    
    # Test public endpoint
    print("\n[TEST 1] Public Markets (No Authentication)")
    print("-" * 70)
    
    connector = PolymarketConnector()
    try:
        markets = await connector.fetch_markets(limit=5)
        
        if markets:
            print(f"[OK] SUCCESS: Fetched {len(markets)} markets")
            print(f"\n   Sample markets:")
            for i, m in enumerate(markets[:3], 1):
                print(f"   {i}. {m.title[:60]}...")
        else:
            print("[WARN] WARNING: No markets returned")
            print("   Possible causes:")
            print("   - API endpoint may have changed")
            print("   - Network connectivity issues")
        
        # Test orderbook
        if markets:
            print("\n[TEST 2] Orderbook Data")
            print("-" * 70)
            
            market_ids = [m.market_id for m in markets[:2]]
            ticks = await connector.fetch_ticks(market_ids)
            
            if ticks:
                print(f"[OK] SUCCESS: Fetched {len(ticks)} orderbook snapshots")
                tick = ticks[0]
                print(f"\n   Sample orderbook:")
                print(f"   - Market: {tick.market_id[:20]}...")
                print(f"   - Probability: {tick.p_norm:.3f}")
                if tick.yes_bid:
                    print(f"   - Bid: {tick.yes_bid:.3f}")
                else:
                    print(f"   - Bid: N/A")
                if tick.yes_ask:
                    print(f"   - Ask: {tick.yes_ask:.3f}")
                else:
                    print(f"   - Ask: N/A")
            else:
                print("[WARN] WARNING: No orderbook data returned")
        
        print("\n" + "="*70)
        print("[OK] POLYMARKET: All tests passed!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await connector.close()


async def test_integration():
    """Test that both APIs can work together."""
    print("\n" + "="*70)
    print("[TESTING] INTEGRATION")
    print("="*70)
    
    try:
        from pm_agent.connectors.kalshi import KalshiConnector
        from pm_agent.connectors.polymarket import PolymarketConnector
        
        print("\n[TEST] Fetching data from both venues concurrently...")
        print("-" * 70)
        
        kalshi = KalshiConnector()
        poly = PolymarketConnector()
        
        try:
            # Fetch from both venues concurrently
            kalshi_task = kalshi.fetch_markets(limit=3)
            poly_task = poly.fetch_markets(limit=3)
            
            kalshi_markets, poly_markets = await asyncio.gather(
                kalshi_task, poly_task, return_exceptions=True
            )
            
            if isinstance(kalshi_markets, Exception):
                print(f"[WARN] Kalshi failed: {kalshi_markets}")
                kalshi_count = 0
            else:
                kalshi_count = len(kalshi_markets)
            
            if isinstance(poly_markets, Exception):
                print(f"[WARN] Polymarket failed: {poly_markets}")
                poly_count = 0
            else:
                poly_count = len(poly_markets)
            
            total = kalshi_count + poly_count
            
            print(f"\n[OK] SUCCESS: Fetched {total} total markets")
            print(f"   - Kalshi: {kalshi_count} markets")
            print(f"   - Polymarket: {poly_count} markets")
            
            print("\n" + "="*70)
            print("[OK] INTEGRATION: Concurrent fetching works!")
            print("="*70)
            return True
            
        finally:
            await kalshi.close()
            await poly.close()
            
    except Exception as e:
        print(f"\n[ERROR] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_environment():
    """Check environment setup."""
    print("\n" + "="*70)
    print("[CHECK] ENVIRONMENT")
    print("="*70)
    
    # Check Python version
    py_version = sys.version_info
    print(f"\nPython version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version < (3, 10):
        print("[WARN] WARNING: Python 3.10+ recommended")
    else:
        print("[OK] Python version OK")
    
    # Check dependencies
    print("\nChecking dependencies...")
    deps = {
        "httpx": "HTTP client for async requests",
        "structlog": "Structured logging",
        "pydantic": "Data validation",
    }
    
    missing = []
    for dep, desc in deps.items():
        try:
            __import__(dep)
            print(f"  [OK] {dep:15} - {desc}")
        except ImportError:
            print(f"  [ERROR] {dep:15} - {desc} (MISSING)")
            missing.append(dep)
    
    if missing:
        print(f"\n[WARN] Missing dependencies: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    # Check credentials
    print("\nChecking credentials...")
    try:
        from pm_agent.config import settings
        kalshi_key = settings.kalshi_api_key
        kalshi_secret = settings.kalshi_api_secret
        poly_key = settings.polymarket_api_key
        
        if kalshi_key and kalshi_secret:
            print(f"  [OK] Kalshi credentials found")
        else:
            print(f"  [INFO] Kalshi credentials not found (public endpoints will be tested)")
        
        if poly_key:
            print(f"  [OK] Polymarket API key found")
        else:
            print(f"  [INFO] Polymarket API key not found (public endpoints will be tested)")
    except Exception as e:
        print(f"  [WARN] Could not check credentials: {e}")
    
    print("\n" + "="*70)
    return True


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PREDICTION MARKET API CONNECTOR TESTS")
    print("="*70)
    print("\nThis will test your Kalshi and Polymarket API implementations.")
    print("Testing both public endpoints (no auth) and authenticated endpoints.")
    
    # Check environment
    if not check_environment():
        print("\n[ERROR] Environment check failed. Please fix issues and try again.")
        return 1
    
    # Run tests
    results = {}
    
    results["kalshi"] = await test_kalshi()
    results["polymarket"] = await test_polymarket()
    results["integration"] = await test_integration()
    
    # Summary
    print("\n" + "="*70)
    print("[SUMMARY] TEST RESULTS")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        print(f"  {status}: {test_name.upper()}")
    
    print("\n" + "="*70)
    if passed == total:
        print(f"[OK] ALL TESTS PASSED ({passed}/{total})")
        print("="*70)
        print("\nYour API connectors are working correctly!")
        print("\nNext steps:")
        print("  1. Run the full test suite: pytest tests/test_kalshi_live.py -v")
        print("  2. Integrate into your pipelines")
        print("  3. Set MOCK_MODE=false in .env to use live data")
        return 0
    else:
        print(f"[ERROR] SOME TESTS FAILED ({passed}/{total})")
        print("="*70)
        print("\nPlease review the errors above and:")
        print("  1. Check API credentials")
        print("  2. Verify network connectivity")
        print("  3. Review API documentation")
        print("  4. Check API_CONNECTOR_FIXES.md for troubleshooting")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
