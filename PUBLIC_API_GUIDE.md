# Using Public APIs (No Authentication Required)

This project supports **free public endpoints** from Kalshi and Polymarket that don't require API keys or paid plans.

## Quick Start

1. **Set `MOCK_MODE=false` in `.env`** (no API keys needed!)
2. **Run the pipelines** - they'll automatically use public endpoints

```powershell
# Edit .env
MOCK_MODE=false
# Leave API keys empty - they're optional!

# Run pipelines
$env:PYTHONPATH="src"
.\.venv_win\Scripts\python.exe -m pipelines.ingest_markets
.\.venv_win\Scripts\python.exe -m pipelines.ingest_ticks
```

## What Works Without Authentication

### Kalshi
- ✅ **Public market data endpoints**: `/markets`, `/markets/{ticker}/orderbook`
- ✅ **Market listings**: Fetch all active markets
- ✅ **Orderbook data**: Best bid/ask prices
- ❌ **Trading endpoints**: Require authentication
- ❌ **Account info**: Requires authentication

**Rate Limits**: ~100 requests/minute (automatically handled by rate limiter)

### Polymarket
- ✅ **CLOB GraphQL endpoints**: Public read-only access
- ✅ **Market listings**: Fetch all active markets
- ✅ **Orderbook data**: Best bid/ask prices
- ❌ **Trading endpoints**: Require authentication
- ❌ **Account info**: Requires authentication

**Rate Limits**: ~1000 requests/minute (automatically handled by rate limiter)

## Rate Limiting & Retry Logic

The connectors automatically handle:
- **Rate limiting**: Token bucket algorithm prevents exceeding API limits
- **Retry with backoff**: Exponential backoff on transient failures
- **Error handling**: Graceful degradation on API errors

## When to Use API Keys

API keys are **only needed** if you want to:
- Place trades
- Access account information
- Use private endpoints

For **data collection and analysis**, public endpoints are sufficient!

## Configuration

### Option 1: Public Endpoints (No Auth)
```env
MOCK_MODE=false
# Leave API keys empty
KALSHI_API_KEY=
KALSHI_API_SECRET=
POLYMARKET_API_KEY=
```

### Option 2: Authenticated Endpoints
```env
MOCK_MODE=false
KALSHI_API_KEY=your_key_here
KALSHI_API_SECRET=your_secret_here
POLYMARKET_API_KEY=your_key_here
```

### Option 3: Mock Data (Development)
```env
MOCK_MODE=true
# API keys ignored in mock mode
```

## Engineering Considerations

As you noted, the main "cost" isn't API fees—it's engineering around:

1. **Rate Limits**: Handled automatically by `RateLimiter` class
2. **Data Quality**: Validators check probability ranges, timestamps, etc.
3. **Market → Ticker Mapping**: Handled by `market_entity_map.yml`

## Testing Public Endpoints

```powershell
# Test Kalshi public endpoint
$env:PYTHONPATH="src"
.\.venv_win\Scripts\python.exe -c "
import asyncio
from pm_agent.connectors.kalshi import KalshiConnector

async def test():
    k = KalshiConnector()  # No API keys!
    markets = await k.fetch_markets(limit=5)
    print(f'Fetched {len(markets)} markets')
    await k.close()

asyncio.run(test())
"
```

## Troubleshooting

### "Rate limit exceeded"
- The rate limiter should handle this automatically
- If you see this, the API may have stricter limits than expected
- Solution: Reduce request frequency or add delays

### "Authentication required"
- Check that you're using public endpoints (markets, orderbook)
- Some endpoints may require auth even if documented as public
- Solution: Check API documentation for endpoint requirements

### "No data returned"
- Public endpoints may have different response formats
- Check logs for API errors
- Solution: Verify endpoint URLs and response parsing

## References

- **Kalshi API Docs**: https://trading-api.kalshi.com/
- **Polymarket CLOB**: https://clob.polymarket.com/
- **Rate Limiting**: See `src/pm_agent/connectors/rate_limit.py`

