"""Test script to verify live API endpoints are working."""
import asyncio
from pm_agent.connectors.kalshi import KalshiConnector
from pm_agent.connectors.polymarket import PolymarketConnector
from pm_agent.config import settings

async def test_kalshi():
    print("=" * 60)
    print("Testing Kalshi Public API (no auth required)")
    print("=" * 60)
    k = KalshiConnector()  # No API keys needed for public endpoints
    try:
        markets = await k.fetch_markets(limit=5)
        print(f"[OK] Successfully fetched {len(markets)} markets from Kalshi")
        if markets:
            print("\nSample markets:")
            for i, m in enumerate(markets[:3], 1):
                print(f"  {i}. {m.market_id}: {m.title[:70]}...")
                print(f"     Status: {m.status}, Venue: {m.venue_id}")
        else:
            print("[WARN] No markets returned (API may be down or rate limited)")
    except Exception as e:
        print(f"[ERROR] Error fetching from Kalshi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await k.close()

async def test_polymarket():
    print("\n" + "=" * 60)
    print("Testing Polymarket Public API (no auth required)")
    print("=" * 60)
    p = PolymarketConnector()  # No API key needed for public endpoints
    try:
        markets = await p.fetch_markets(limit=5)
        print(f"[OK] Successfully fetched {len(markets)} markets from Polymarket")
        if markets:
            print("\nSample markets:")
            for i, m in enumerate(markets[:3], 1):
                print(f"  {i}. {m.market_id}: {m.title[:70]}...")
                print(f"     Status: {m.status}, Venue: {m.venue_id}")
        else:
            print("[WARN] No markets returned (API may be down or rate limited)")
    except Exception as e:
        print(f"[ERROR] Error fetching from Polymarket: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await p.close()

async def main():
    print(f"\nCurrent Configuration:")
    print(f"  MOCK_MODE: {settings.mock_mode}")
    print(f"  Kalshi API Key: {'SET' if settings.kalshi_api_key else 'NOT SET (using public endpoints)'}")
    print(f"  Polymarket API Key: {'SET' if settings.polymarket_api_key else 'NOT SET (using public endpoints)'}")
    print()
    
    await test_kalshi()
    await test_polymarket()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    if settings.mock_mode:
        print("[WARN] MOCK_MODE is True - pipelines will use mock data")
        print("   Set MOCK_MODE=false in .env to use live APIs")
    else:
        print("[OK] MOCK_MODE is False - pipelines will use live APIs")
        print("   Public endpoints work without API keys!")

if __name__ == "__main__":
    asyncio.run(main())

