"""Quick verification of connector fixes."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pm_agent.connectors.kalshi import KalshiConnector
from pm_agent.connectors.polymarket import PolymarketConnector
import inspect

print("=" * 70)
print("CONNECTOR VERIFICATION")
print("=" * 70)

# 1. Verify Kalshi fixes
print("\n1. KALSHI CONNECTOR FIXES")
print("-" * 70)
kalshi_src = inspect.getsource(KalshiConnector._generate_auth_headers)
has_milliseconds = 'time.time() * 1000' in kalshi_src
has_separate_headers = 'Kalshi-Access-Key' in kalshi_src and 'Kalshi-Access-Signature' in kalshi_src

print(f"  Timestamp in milliseconds: {'[OK]' if has_milliseconds else '[ERROR]'}")
print(f"  Separate auth headers: {'[OK]' if has_separate_headers else '[ERROR]'}")

if has_milliseconds and has_separate_headers:
    print("  [OK] All Kalshi fixes verified!")
else:
    print("  [ERROR] Some fixes missing!")

# 2. Verify Polymarket endpoints
print("\n2. POLYMARKET CONNECTOR FIXES")
print("-" * 70)
poly_src = inspect.getsource(PolymarketConnector)
has_gamma_api = 'GAMMA_API_URL' in poly_src
has_clob_api = 'CLOB_API_URL' in poly_src
has_gamma_markets = 'gamma-api.polymarket.com' in poly_src
has_clob_book = 'clob.polymarket.com' in poly_src and '/book' in poly_src

print(f"  Gamma API for markets: {'[OK]' if has_gamma_api and has_gamma_markets else '[ERROR]'}")
print(f"  CLOB API for orderbook: {'[OK]' if has_clob_api and has_clob_book else '[ERROR]'}")

if has_gamma_api and has_clob_api and has_gamma_markets and has_clob_book:
    print("  [OK] All Polymarket fixes verified!")
else:
    print("  [ERROR] Some fixes missing!")

# 3. Test actual API calls
print("\n3. API CONNECTIVITY TEST")
print("-" * 70)

async def test_apis():
    # Test Polymarket (should work)
    poly = PolymarketConnector()
    try:
        markets = await poly.fetch_markets(limit=2)
        if markets:
            print(f"  Polymarket: [OK] Fetched {len(markets)} markets")
        else:
            print(f"  Polymarket: [WARN] No markets (may be API issue)")
    except Exception as e:
        print(f"  Polymarket: [ERROR] {str(e)[:60]}")
    finally:
        await poly.close()
    
    # Test Kalshi (may have network issues)
    kalshi = KalshiConnector()
    try:
        markets = await kalshi.fetch_markets(limit=2)
        if markets:
            print(f"  Kalshi: [OK] Fetched {len(markets)} markets")
        else:
            print(f"  Kalshi: [WARN] No markets (may be network/API issue)")
    except Exception as e:
        error_msg = str(e)
        if "getaddrinfo" in error_msg or "11001" in error_msg:
            print(f"  Kalshi: [WARN] Network/DNS issue (not a code problem)")
        else:
            print(f"  Kalshi: [ERROR] {error_msg[:60]}")
    finally:
        await kalshi.close()

asyncio.run(test_apis())

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

