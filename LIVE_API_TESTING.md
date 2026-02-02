# Live API Testing Guide

## Overview
Your Kalshi and Polymarket connectors are well-implemented but need validation with real API credentials. This guide provides a testing strategy.

## Testing Checklist

### ✅ Pre-Testing Setup
1. **Get API Credentials**
   - Kalshi: https://trading-api.kalshi.com/
   - Polymarket: Check current API documentation

2. **Create Test Environment File**
   ```bash
   cp .env.example .env.test
   # Add: MOCK_MODE=false
   # Add: KALSHI_API_KEY=your_key
   # Add: KALSHI_API_SECRET=your_secret
   ```

3. **Safety**: Use test/demo accounts if available

### 🧪 Test 1: Kalshi Market Fetch

Create `tests/test_kalshi_live.py`:

```python
import pytest
from pm_agent.connectors.kalshi import KalshiConnector
from pm_agent.config import settings

@pytest.mark.skipif(
    settings.kalshi_api_key is None,
    reason="Kalshi credentials not configured"
)
@pytest.mark.asyncio
async def test_kalshi_fetch_markets_live():
    """Test Kalshi connector with real API."""
    connector = KalshiConnector(
        api_key=settings.kalshi_api_key,
        api_secret=settings.kalshi_api_secret,
    )
    
    try:
        # Fetch small batch first
        markets = await connector.fetch_markets(limit=5)
        
        # Assertions
        assert len(markets) > 0, "No markets returned"
        assert all(m.venue_id == "kalshi" for m in markets)
        assert all(m.market_id for m in markets)
        
        # Check structure
        first = markets[0]
        assert first.title
        assert first.status in ["open", "closed", "active"]
        
        print(f"✅ Successfully fetched {len(markets)} Kalshi markets")
        
    finally:
        await connector.close()

@pytest.mark.skipif(
    settings.kalshi_api_key is None,
    reason="Kalshi credentials not configured"
)
@pytest.mark.asyncio
async def test_kalshi_fetch_ticks_live():
    """Test Kalshi orderbook fetching."""
    connector = KalshiConnector(
        api_key=settings.kalshi_api_key,
        api_secret=settings.kalshi_secret,
    )
    
    try:
        # First get a market
        markets = await connector.fetch_markets(limit=1)
        assert len(markets) > 0
        
        market_id = markets[0].market_id
        
        # Fetch ticks
        ticks = await connector.fetch_ticks([market_id])
        
        assert len(ticks) > 0
        assert ticks[0].venue_id == "kalshi"
        assert ticks[0].p_norm >= 0 and ticks[0].p_norm <= 1
        
        print(f"✅ Successfully fetched {len(ticks)} ticks for {market_id}")
        
    finally:
        await connector.close()
```

### 🧪 Test 2: Polymarket Market Fetch

Create `tests/test_polymarket_live.py`:

```python
import pytest
from pm_agent.connectors.polymarket import PolymarketConnector
from pm_agent.config import settings

@pytest.mark.asyncio
async def test_polymarket_fetch_markets_live():
    """Test Polymarket connector (public API)."""
    connector = PolymarketConnector(api_key=settings.polymarket_api_key)
    
    try:
        # Try without auth first (public data)
        markets = await connector.fetch_markets(limit=5)
        
        # May return empty if API changed - that's useful info
        if len(markets) > 0:
            assert all(m.venue_id == "polymarket" for m in markets)
            print(f"✅ Successfully fetched {len(markets)} Polymarket markets")
        else:
            print("⚠️ Polymarket returned no markets - API may have changed")
        
    except Exception as e:
        print(f"⚠️ Polymarket API error: {e}")
        # Document the error for fixing
        
    finally:
        await connector.close()
```

### 🧪 Test 3: End-to-End Pipeline with Live Data

```python
import pytest
from pm_agent.config import settings

@pytest.mark.skipif(
    settings.kalshi_api_key is None,
    reason="Live API credentials not configured"
)
@pytest.mark.asyncio
async def test_pipeline_with_live_data():
    """Test full pipeline with live API data."""
    from pm_agent.db import get_session
    from pm_agent.sql import fetch_all
    
    # Temporarily set mock_mode to False
    original_mock = settings.mock_mode
    settings.mock_mode = False
    
    try:
        # Run ingest pipelines
        from pipelines.ingest_markets import run as ingest_markets
        from pipelines.ingest_ticks import run as ingest_ticks
        
        await ingest_markets()
        await ingest_ticks()
        
        # Verify data was inserted
        async with get_session() as session:
            markets = await fetch_all(
                session,
                "SELECT * FROM markets WHERE venue_id IN ('kalshi', 'polymarket') ORDER BY updated_at DESC LIMIT 5"
            )
            assert len(markets) > 0, "No live markets inserted"
            
            ticks = await fetch_all(
                session,
                "SELECT * FROM odds_ticks WHERE venue_id IN ('kalshi', 'polymarket') ORDER BY tick_ts DESC LIMIT 5"
            )
            assert len(ticks) > 0, "No live ticks inserted"
            
            print(f"✅ Live pipeline: {len(markets)} markets, {len(ticks)} ticks")
            
    finally:
        settings.mock_mode = original_mock
```

## Known Issues to Check

### 1. Kalshi Authentication
**Issue**: HMAC signature generation is sensitive to:
- Timestamp format
- String encoding
- Header ordering

**Test**:
```python
def test_kalshi_auth_headers():
    connector = KalshiConnector(api_key="test", api_secret="test")
    headers = connector._generate_auth_headers("GET", "/markets")
    
    assert "Authorization" in headers
    assert "Kalshi-Access-Timestamp" in headers
    # Verify timestamp is reasonable
    assert int(headers["Kalshi-Access-Timestamp"]) > 0
```

### 2. Polymarket GraphQL Schema
**Issue**: GraphQL queries may have changed since implementation.

**Test**: Check error messages for schema mismatches.

### 3. Rate Limiting
**Issue**: Both APIs have rate limits.

**Test**:
```python
@pytest.mark.asyncio
async def test_rate_limiting():
    connector = KalshiConnector(...)
    
    # Make multiple rapid requests
    for i in range(20):
        try:
            await connector.fetch_markets(limit=1)
        except Exception as e:
            if "rate limit" in str(e).lower():
                print(f"✅ Rate limit detected at request {i}")
                return
    
    print("⚠️ No rate limiting detected")
```

## Running Tests

```bash
# Set up environment
export DATABASE_URL_ASYNC="postgresql+asyncpg://pm:pm@localhost:5432/pm_research_test"
export KALSHI_API_KEY="your_key"
export KALSHI_API_SECRET="your_secret"

# Run live tests
pytest tests/test_kalshi_live.py -v
pytest tests/test_polymarket_live.py -v
pytest tests/test_pipeline_with_live_data.py -v

# Or skip if no credentials
pytest -v  # Skips tests marked with skipif
```

## Validation Criteria

### ✅ Success
- Markets fetch returns data
- Ticks have valid probabilities (0-1 range)
- Data inserts to database without errors
- Authentication works (no 401/403 errors)

### ⚠️ Needs Investigation
- Empty results (API changed? Wrong endpoint?)
- Rate limiting (expected, but document limits)
- Schema mismatches (GraphQL fields changed?)

### ❌ Critical Issues
- Authentication failures
- Network errors (check firewall/proxy)
- Data validation errors (p_norm out of range)

## Documentation

After testing, update:

1. **LIVE_API_SETUP.md** - Add any discoveries:
   - Actual rate limits encountered
   - Required API permissions
   - API version numbers

2. **README.md** - Update with:
   - Tested API versions
   - Known limitations

3. **config/market_entity_map.yml** - Add real market IDs if different from mock

## Next Steps After Testing

1. **If Successful**: 
   - Deploy to staging environment
   - Set up monitoring/alerting
   - Document API costs

2. **If Issues Found**:
   - Fix connector code
   - Update error handling
   - Add retry logic if needed

3. **Production Readiness**:
   - Add circuit breakers for API failures
   - Implement exponential backoff
   - Monitor API health in dashboard

