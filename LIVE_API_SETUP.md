# Live Prediction Market API Setup

This guide explains how to connect to **live** prediction market APIs (Kalshi and Polymarket) instead of using mock data.

## Quick Start

1. **Get API credentials** (see below)
2. **Create `.env` file** from `env.template`
3. **Set `MOCK_MODE=false`** in `.env`
4. **Add your API keys** to `.env`
5. **Run pipelines** - they'll now fetch live data!

## API Credentials

### Kalshi

1. Go to [Kalshi Trading API](https://trading-api.kalshi.com/)
2. Sign up / Log in
3. Generate API key and secret
4. Add to `.env`:
   ```
   KALSHI_API_KEY=your_key_here
   KALSHI_API_SECRET=your_secret_here
   ```

**Note**: Kalshi API requires authentication for most endpoints.

### Polymarket

1. Go to [Polymarket](https://polymarket.com/)
2. Some endpoints work without authentication (public data)
3. For authenticated access, check Polymarket's developer docs
4. Add to `.env` (optional):
   ```
   POLYMARKET_API_KEY=your_key_here
   ```

**Note**: Polymarket uses GraphQL. Public markets may be accessible without API key.

## Configuration

### Enable Live Mode

In your `.env` file:

```bash
# Switch from mock to live
MOCK_MODE=false

# Add your credentials
KALSHI_API_KEY=your_key
KALSHI_API_SECRET=your_secret
POLYMARKET_API_KEY=your_key  # Optional
```

### Verify Setup

```bash
# Check config loads correctly
$env:PYTHONPATH="src"
python -c "from pm_agent.config import settings; print(f'Mock mode: {settings.mock_mode}'); print(f'Kalshi key set: {settings.kalshi_api_key is not None}')"
```

### Create `.env` from template (recommended)

This repo intentionally does **not** commit `.env` (secrets). Use the committed template:

```bash
Copy-Item env.template .env
```

## Running with Live Data

Once configured, run pipelines normally:

```bash
# Ingest live markets
$env:PYTHONPATH="src"
python -m pipelines.ingest_markets

# Ingest live ticks (orderbook data)
$env:PYTHONPATH="src"
python -m pipelines.ingest_ticks

# Or run full pipeline
python -m pipelines.run_all
```

## What Gets Fetched

### Markets (`ingest_markets`)
- **Kalshi**: Active markets via `/markets` endpoint
- **Polymarket**: Active markets via GraphQL `markets` query
- Markets are stored in PostgreSQL `markets` table

### Ticks (`ingest_ticks`)
- **Kalshi**: Orderbook data via `/markets/{ticker}/orderbook`
- **Polymarket**: Orderbook via GraphQL `orderbook` query
- Best bid/ask prices extracted and normalized
- Stored in PostgreSQL `odds_ticks` table

## Rate Limiting

Both APIs have rate limits:

- **Kalshi**: Check your API plan limits
- **Polymarket**: Public endpoints may have stricter limits

**Recommendation**: 
- Don't run `ingest_ticks` too frequently (every 5-10 minutes is reasonable)
- Use `ingest_markets` less frequently (once per hour or daily)

## Error Handling

The connectors handle errors gracefully:

- **API failures**: Logged but don't crash the pipeline
- **Missing credentials**: Falls back to empty results (logged)
- **Network errors**: Retried automatically by `httpx`

Check logs for:
```
{"event": "kalshi_fetch_markets_error", "error": "..."}
{"event": "polymarket_fetch_markets_error", "error": "..."}
```

## Switching Back to Mock

If you want to use mock data again:

```bash
# In .env
MOCK_MODE=true
```

Or just remove the API keys - the system will default to mock mode.

## Troubleshooting

### "No markets fetched"

**Possible causes**:
1. API credentials missing or invalid
2. API endpoint changed
3. Network/firewall blocking requests

**Fix**:
- Verify credentials in `.env`
- Check API status pages
- Test with `curl` or Postman

### "Authentication failed"

**Kalshi**:
- Verify API key and secret are correct
- Check timestamp sync (Kalshi uses timestamp-based auth)
- Ensure API key hasn't expired

**Polymarket**:
- Try without API key first (public data)
- Verify API key format if using authenticated endpoints

### "GraphQL query failed"

**Polymarket**:
- Check query syntax matches current API version
- Verify market IDs are valid
- Some fields may require authentication

## Cost Considerations

- **Kalshi**: May charge per API call (check your plan)
- **Polymarket**: Public data is usually free, authenticated may have costs

**Recommendation**: Start with mock data for development, switch to live for production/demos.

## Next Steps

1. ✅ Get API credentials
2. ✅ Configure `.env`
3. ✅ Test with `ingest_markets` first
4. ✅ Verify data in database
5. ✅ Run full pipeline
6. ✅ Check dashboard shows live data

## Support

- **Kalshi API Docs**: https://trading-api.kalshi.com/docs
- **Polymarket**: Check their developer documentation
- **Issues**: Check logs for specific error messages

---

**Remember**: Always test with mock data first, then switch to live APIs once everything works!

