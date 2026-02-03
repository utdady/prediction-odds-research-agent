"""Quick API validation test - Run this FIRST (no credentials needed)."""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pm_agent.connectors.kalshi import KalshiConnector
from pm_agent.connectors.polymarket import PolymarketConnector
from pm_agent.config import settings
import structlog

log = structlog.get_logger(__name__)


async def test_environment():
    """Check environment configuration."""
    print("=" * 70)
    print("1. ENVIRONMENT CHECK")
    print("=" * 70)
    print(f"  MOCK_MODE: {settings.mock_mode}")
    print(f"  Kalshi API Key: {'SET' if settings.kalshi_api_key else 'NOT SET (using public)'}")
    print(f"  Polymarket API Key: {'SET' if settings.polymarket_api_key else 'NOT SET (using public)'}")
    print()


async def test_kalshi_public():
    """Test Kalshi public endpoints."""
    print("=" * 70)
    print("2. KALSHI PUBLIC ENDPOINTS")
    print("=" * 70)
    connector = KalshiConnector()  # No credentials
    
    try:
        print("  Testing /markets endpoint...")
        markets = await connector.fetch_markets(limit=5)
        
        if markets:
            print(f"  [OK] Successfully fetched {len(markets)} markets")
            print(f"  Sample: {markets[0].market_id} - {markets[0].title[:60]}...")
            print(f"  Status: {markets[0].status}")
        else:
            print("  [WARN] No markets returned (API may be down or rate limited)")
            
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await connector.close()
    print()


async def test_polymarket_public():
    """Test Polymarket public endpoints."""
    print("=" * 70)
    print("3. POLYMARKET PUBLIC ENDPOINTS")
    print("=" * 70)
    connector = PolymarketConnector()  # No credentials
    
    try:
        print("  Testing GraphQL markets endpoint...")
        markets = await connector.fetch_markets(limit=5)
        
        if markets:
            print(f"  [OK] Successfully fetched {len(markets)} markets")
            print(f"  Sample: {markets[0].market_id} - {markets[0].title[:60]}...")
            print(f"  Status: {markets[0].status}")
        else:
            print("  [WARN] No markets returned (API may be down or rate limited)")
            
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await connector.close()
    print()


async def test_data_quality():
    """Validate data quality."""
    print("=" * 70)
    print("4. DATA QUALITY VALIDATION")
    print("=" * 70)
    
    kalshi = KalshiConnector()
    poly = PolymarketConnector()
    
    try:
        k_markets = await kalshi.fetch_markets(limit=10)
        p_markets = await poly.fetch_markets(limit=10)
        
        issues = []
        
        # Check Kalshi
        if k_markets:
            for m in k_markets:
                if not m.market_id:
                    issues.append(f"Kalshi: Missing market_id")
                if not m.title:
                    issues.append(f"Kalshi: Missing title for {m.market_id}")
        else:
            issues.append("Kalshi: No markets returned")
        
        # Check Polymarket
        if p_markets:
            for m in p_markets:
                if not m.market_id:
                    issues.append(f"Polymarket: Missing market_id")
                if not m.title:
                    issues.append(f"Polymarket: Missing title for {m.market_id}")
        else:
            issues.append("Polymarket: No markets returned")
        
        if issues:
            print("  [WARN] Data quality issues found:")
            for issue in issues[:5]:  # Show first 5
                print(f"    - {issue}")
        else:
            print("  [OK] Data quality checks passed")
            
    except Exception as e:
        print(f"  [ERROR] Validation failed: {e}")
    finally:
        await kalshi.close()
        await poly.close()
    print()


async def test_concurrent_requests():
    """Test concurrent requests (rate limiting)."""
    print("=" * 70)
    print("5. CONCURRENT REQUESTS TEST")
    print("=" * 70)
    
    connector = KalshiConnector()
    
    try:
        print("  Making 3 concurrent requests...")
        tasks = [connector.fetch_markets(limit=3) for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success = sum(1 for r in results if not isinstance(r, Exception))
        errors = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"  [OK] Success: {success}/3")
        if errors > 0:
            print(f"  [WARN] Errors: {errors}/3 (may be rate limiting)")
            
    except Exception as e:
        print(f"  [ERROR] Concurrent test failed: {e}")
    finally:
        await connector.close()
    print()


async def main():
    """Run all quick tests."""
    print("\n" + "=" * 70)
    print("QUICK API VALIDATION TEST")
    print("=" * 70)
    print("This test validates public endpoints (no credentials needed)")
    print("=" * 70)
    print()
    
    await test_environment()
    await test_kalshi_public()
    await test_polymarket_public()
    await test_data_quality()
    await test_concurrent_requests()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("[OK] Quick validation complete!")
    print("Next steps:")
    print("  1. Get Kalshi credentials (optional)")
    print("  2. Run comprehensive tests: pytest tests/test_kalshi_live.py -v")
    print("  3. Replace connectors if fixes are needed")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

