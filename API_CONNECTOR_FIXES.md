# API Connector Fixes & Testing Guide

## 📋 Overview

This document covers the fixes applied to Kalshi and Polymarket connectors, along with comprehensive testing procedures.

## 🔧 Fixed Issues

### Kalshi Connector Fixes

| Issue | Original | Fixed | Impact |
|-------|----------|-------|--------|
| **HMAC Auth** | `base64(key:sig)` in Authorization | Separate headers: `Kalshi-Access-Key` and `Kalshi-Access-Signature` | ✅ Auth works correctly |
| **Timestamp** | `int(time.time())` (seconds) | `int(time.time() * 1000)` (milliseconds) | ✅ Correct format |
| **Error Messages** | Generic exceptions | Detailed debugging info | 🔧 Easier troubleshooting |

### Polymarket Connector Fixes

| Issue | Original | Fixed | Impact |
|-------|----------|-------|--------|
| **API Endpoint** | GraphQL only | Gamma API + CLOB | ✅ Current API structure |
| **Error Messages** | Generic exceptions | Detailed debugging info | 🔧 Easier troubleshooting |

## 🚀 Quick Start (Next 15 Minutes)

### Step 1: Run Quick Test (No Credentials Needed!)

```bash
# Set PYTHONPATH
$env:PYTHONPATH="src"

# Run quick test
python tests/quick_test_apis.py
```

**Expected Result:** Both APIs should work with public data!

**What it tests:**
- ✅ Environment configuration
- ✅ Public endpoints (both APIs)
- ✅ Data quality validation
- ✅ Concurrent requests (rate limiting)

### Step 2: Get Credentials (Optional but Recommended)

#### Kalshi (5 minutes):
1. Go to https://kalshi.com/
2. Settings → API Access
3. Generate key → Save `member_id` and `private_key`
4. Add to `.env`:
   ```
   KALSHI_API_KEY=your_member_id
   KALSHI_API_SECRET=your_private_key
   ```

#### Polymarket:
- **Public CLOB API works without credentials!**
- Only need API key for trading endpoints

### Step 3: Test with Credentials

```bash
# Set environment variables (or use .env file)
$env:KALSHI_API_KEY="your_member_id"
$env:KALSHI_API_SECRET="your_private_key"

# Run comprehensive tests
pytest tests/test_kalshi_live.py -v -s
pytest tests/test_polymarket_live.py -v -s
```

### Step 4: Verify Connectors Are Fixed

The fixes are already applied in:
- `src/pm_agent/connectors/kalshi.py`
- `src/pm_agent/connectors/polymarket.py`

No need to replace files - they're already updated!

## 📊 API Comparison

### Kalshi vs Polymarket

| Feature | Kalshi | Polymarket |
|---------|--------|------------|
| **Public Endpoints** | ✅ Yes (`/markets`, `/orderbook`) | ✅ Yes (CLOB GraphQL) |
| **Auth Required** | ❌ No (for public data) | ❌ No (for public data) |
| **Rate Limits** | ~100 req/min | ~1000 req/min |
| **Data Format** | REST JSON | GraphQL |
| **Orderbook** | ✅ Yes | ✅ Yes |
| **Market Listings** | ✅ Yes | ✅ Yes |

## 🎯 What to Expect

### Public Endpoints (No Credentials):
- ✅ **Kalshi**: Fetch 100s of markets
- ✅ **Polymarket**: Fetch 100s of markets + orderbooks
- ✅ **Rate limiting**: Automatically handled

### Authenticated Endpoints (With Credentials):
- ✅ Same data access
- ✅ May have higher rate limits
- ✅ Access to account data (if needed)

## 🔥 Priority Action Items

### 🔴 This Week (Critical):
1. ✅ Run `quick_test_apis.py` (5 min)
2. ✅ Get Kalshi credentials (5 min) - Optional
3. ✅ Run full test suite (10 min)
4. ✅ Verify connectors are working

### 🟠 Next Week (High):
1. Update `ingest_markets.py` to use fixed connectors
2. Run full pipeline with live data
3. Validate data quality in database
4. Set up monitoring

### 🟡 Next Month (Medium):
1. Deploy to staging
2. Implement alerts
3. Production deployment

## 💡 Pro Tips

1. **Start with public endpoints** - They work great for research!
2. **Test incrementally** - Run `quick_test_apis.py` first
3. **Monitor rate limits** - Both APIs are generous but watch for 429 errors
4. **Cache aggressively** - No need to fetch markets every minute
5. **Log everything** - Structured logs are your friend

## 🐛 Common Issues & Solutions

### Issue: "getaddrinfo failed" (Kalshi)
**Solution:** Check network connectivity. The endpoint `https://api.kalshi.co/trade-api/v2` should be accessible.

### Issue: "404 Not Found" (Polymarket)
**Solution:** The GraphQL endpoint may have changed. Check Polymarket's latest API documentation.

### Issue: "Authentication failed" (Kalshi)
**Solution:** 
1. Verify timestamp is in milliseconds (fixed in connector)
2. Check that `Kalshi-Access-Key` and `Kalshi-Access-Signature` headers are separate (fixed in connector)
3. Verify credentials are correct

### Issue: Rate limit exceeded
**Solution:** The rate limiter should handle this automatically. If you see 429 errors, reduce request frequency.

## 📚 Additional Resources

- **Kalshi API Docs**: https://docs.kalshi.com/
- **Polymarket CLOB**: https://clob.polymarket.com/
- **Rate Limiting**: See `src/pm_agent/connectors/rate_limit.py`
- **Public API Guide**: See `PUBLIC_API_GUIDE.md`

