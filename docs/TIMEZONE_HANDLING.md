# Timezone Handling Guide

## Overview

This document explains how timezones are handled throughout the codebase to prevent bugs and ensure consistency.

## Principles

1. **Database Storage**: All timestamps are stored in UTC in PostgreSQL (`TIMESTAMPTZ`)
2. **Application Logic**: Use timezone-aware datetimes when possible
3. **Comparisons**: Always compare timezone-aware datetimes or convert to UTC first
4. **Display**: Convert to user's timezone only for display purposes

## Common Patterns

### ✅ Good: Keep Timezone-Aware

```python
from datetime import datetime, timezone

# Create UTC timestamp
ts = datetime.now(timezone.utc)

# Parse with timezone
ts = pd.to_datetime(row["ts"], utc=True)  # Keeps UTC info

# Compare timezone-aware datetimes
if ts1 > ts2:  # Both are UTC-aware
    ...
```

### ⚠️ Caution: Timezone Conversion

```python
# Only convert to naive if you have a specific reason
ts = pd.Timestamp(row["ts"], tz="UTC").tz_convert(None)

# Document why you're removing timezone:
# "Converting to naive for compatibility with price data index (which is naive)"
```

### ❌ Bad: Mixing Timezone-Aware and Naive

```python
# DON'T do this:
naive_ts = datetime.now()  # No timezone
aware_ts = datetime.now(timezone.utc)  # Has timezone
if naive_ts > aware_ts:  # ❌ TypeError or incorrect comparison
    ...
```

## Codebase-Specific Notes

### Backtest Engine

**File**: `src/pm_agent/backtest/engine.py`

```python
# Line 84: Converts to naive for price index compatibility
ts = pd.Timestamp(s["ts"], tz="UTC").tz_convert(None).normalize()
```

**Rationale**: Price data (`data/prices/*.csv`) uses naive datetimes in the index. Converting signal timestamps to naive ensures they match the price index for lookups.

**Alternative Consideration**: Could keep timezone-aware and convert price data, but current approach is simpler and works for US market hours.

### Feature Building

**File**: `pipelines/build_features.py`

```python
# Line 80: Keeps timezone-aware
df["tick_ts"] = pd.to_datetime(df["tick_ts"], utc=True)
```

**Rationale**: Features are stored with UTC timestamps, so we keep timezone info throughout processing.

### Stock Analysis Agent

**File**: `src/pm_agent/agent/stock_analysis.py`

```python
# Uses parameterized queries - no timezone issues in SQL
cutoff = reference_time - timedelta(days=lookback_days)
params = (cutoff.isoformat(),)  # ISO format includes timezone
```

**Rationale**: PostgreSQL handles `TIMESTAMPTZ` correctly when passed ISO format strings.

## Best Practices

1. **Always use UTC for storage and internal logic**
2. **Document timezone conversions** - explain why you're converting
3. **Use `pd.to_datetime(..., utc=True)`** when reading from database
4. **Use `datetime.now(timezone.utc)`** for current time
5. **Test with different timezones** if your code will run in multiple regions

## Testing Timezone Handling

```python
def test_timezone_aware_comparison():
    """Test that timezone-aware datetimes compare correctly."""
    from datetime import datetime, timezone, timedelta
    
    ts1 = datetime.now(timezone.utc)
    ts2 = ts1 + timedelta(hours=1)
    
    assert ts2 > ts1  # Should work without errors
```

## Common Issues

### Issue 1: "can't compare offset-naive and offset-aware datetime"

**Cause**: Mixing naive and aware datetimes

**Fix**: Convert both to same timezone (preferably UTC)

```python
# Before (broken)
if naive_ts > aware_ts:  # ❌

# After (fixed)
if naive_ts.replace(tzinfo=timezone.utc) > aware_ts:  # ✅
```

### Issue 2: "Timestamp out of range"

**Cause**: Converting timezone-aware to naive can shift the date

**Fix**: Use `.tz_convert(None)` instead of `.replace(tzinfo=None)`

```python
# Before (may shift date)
ts.replace(tzinfo=None)  # ❌

# After (preserves wall-clock time)
ts.tz_convert(None)  # ✅
```

## Migration Notes

If you need to change timezone handling:

1. **Audit all datetime operations** - search for `datetime`, `Timestamp`, `tz_convert`
2. **Test with edge cases** - midnight UTC, DST transitions
3. **Update documentation** - explain the new approach
4. **Consider backward compatibility** - existing data may be in a specific format

## References

- [Python datetime docs](https://docs.python.org/3/library/datetime.html)
- [Pandas timezone docs](https://pandas.pydata.org/docs/user_guide/timeseries.html#time-zone-handling)
- [PostgreSQL timestamp docs](https://www.postgresql.org/docs/current/datatype-datetime.html)

