# Backtest Methodology

## Walk-Forward Analysis

### Why Walk-Forward?

Walk-forward analysis prevents **overfitting** and **lookahead bias** by:
1. Training on past data only
2. Testing on future data
3. Rolling the window forward

### Parameters

- **Training Window**: 252 days (1 trading year)
  - Why: Captures full market cycle (bull, bear, choppy)
  - Trade-off: Longer = more data but slower adaptation

- **Test Window**: 21 days (1 trading month)
  - Why: Short enough to adapt quickly, long enough for statistical significance
  - Trade-off: Shorter = faster adaptation but less reliable metrics

- **Purge Gap**: 5 days
  - Why: Prevents data leakage between train/test
  - Ensures features computed before market close are truly "before"

### Walk-Forward Process

```
Window 1:
  Train: [Day 1-252]
  Purge: [Day 253-257]
  Test:  [Day 258-278]

Window 2:
  Train: [Day 22-273]
  Purge: [Day 274-278]
  Test:  [Day 279-299]

... and so on
```

## Timestamp Alignment

### Critical Rule: No Lookahead Bias

**Features must use data BEFORE market close cutoff:**

```python
# CORRECT: Features use ticks before previous market close
cutoff = get_previous_market_close(current_date)
features = compute_features(ticks[ticks['tick_ts'] < cutoff])
```

**Trades execute on NEXT trading day:**

```python
# CORRECT: Entry is next trading day after signal
signal_date = "2024-01-02"
entry_date = get_next_trading_day(signal_date)  # "2024-01-03"
```

This ensures:
- No information leakage
- Realistic execution timing
- Accurate performance metrics

## Exit Rules

### Stop-Loss: -5%
- **Why**: Limits downside risk
- **Rationale**: If prediction was wrong, cut losses early

### Take-Profit: +10%
- **Why**: Lock in gains
- **Rationale**: Markets can reverse, take profits when available

### Trailing Stop: -3% from peak
- **Why**: Protects profits in volatile markets
- **Rationale**: Allows upside while protecting against reversals

## Cost Model

- **Spread**: 5 bps (0.05%)
- **Slippage**: 5 bps (0.05%)
- **Total**: 10 bps per round trip

**Rationale**: Realistic for liquid markets. Higher costs would reduce returns.

## Position Sizing

- **Equal Weight**: 1/N positions (N = max_positions)
- **Max Positions**: 10
- **Rationale**: Diversification without over-concentration

## Metrics Calculated

1. **Sharpe Ratio**: Risk-adjusted returns
2. **Sortino Ratio**: Downside risk-adjusted returns
3. **Max Drawdown**: Worst peak-to-trough decline
4. **Calmar Ratio**: CAGR / max drawdown
5. **Information Ratio**: Excess return vs benchmark
6. **Omega Ratio**: Probability-weighted gains/losses
7. **Tail Risk**: Skewness and kurtosis

## Validation

- **Out-of-sample testing**: Walk-forward ensures true out-of-sample
- **Multiple time periods**: Tests robustness across market regimes
- **Monte Carlo simulation**: Tests strategy under various scenarios

