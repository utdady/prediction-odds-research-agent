# Feature Engineering Rationale

## Why `delta_p_1d` vs `delta_p_7d`?

### `delta_p_1d` (1-day probability change)
- **Use Case**: Short-term momentum signals
- **Rationale**: Captures recent market sentiment shifts
- **Trade-off**: More noise, but faster signal generation
- **Threshold**: 0.08 (8% change) - filters out minor fluctuations

### `delta_p_7d` (7-day probability change)
- **Use Case**: Longer-term trend confirmation
- **Rationale**: Smoother signal, less noise
- **Trade-off**: Slower to react, but more reliable
- **Not Currently Used**: Could be added for trend confirmation

## Feature Selection

### Core Features

1. **`p_now`** - Current probability
   - Baseline for all comparisons
   - Normalized to [0, 1]

2. **`delta_p_1h`** - 1-hour change
   - Very short-term momentum
   - Captures intraday moves

3. **`delta_p_1d`** - 1-day change
   - Primary signal feature
   - Balances responsiveness vs noise

4. **`rolling_std_p_1d`** - Rolling volatility
   - Measures uncertainty/volatility
   - High volatility = higher risk

5. **`liquidity_score`** - Market liquidity
   - Volume-weighted measure
   - Low liquidity = unreliable prices

6. **`venue_disagreement`** - Cross-venue spread
   - Kalshi vs Polymarket disagreement
   - High disagreement = arbitrage opportunity or uncertainty

7. **`time_to_resolution_days`** - Days until market closes
   - Temporal feature
   - Markets closer to resolution may be more efficient

## Feature Engineering Principles

1. **Normalization**: All probabilities in [0, 1] range
2. **Temporal Alignment**: Features use data **before** market close cutoff (no lookahead)
3. **Missing Data**: Filled with 0.0 (assumes neutral/no signal)
4. **Feature Stability**: Features should be stable over time (not too volatile)

## Future Enhancements

- **Multi-timeframe features**: 1h, 4h, 1d, 7d aggregations
- **Sentiment features**: News sentiment scores
- **Regime features**: Market regime indicators (bull/bear/choppy)
- **Options-implied features**: Comparison to options market pricing

