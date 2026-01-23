# Strategy Explanations

## Why Ensemble?

An ensemble combines multiple strategies to improve robustness and reduce overfitting. Each strategy has different strengths:

### Individual Strategies

1. **RuleStrategyV1** (Weight: 0.3)
   - **Why**: Simple, interpretable, fast
   - **Strengths**: Works well in trending markets with clear signals
   - **Weaknesses**: Misses subtle patterns, no learning

2. **ModelStrategyV1** (Weight: 0.4)
   - **Why**: Learns from historical data, captures non-linear relationships
   - **Strengths**: Adapts to market conditions, probabilistic outputs
   - **Weaknesses**: Requires training data, can overfit

3. **MomentumStrategy** (Weight: 0.2)
   - **Why**: Captures short-term momentum effects
   - **Strengths**: Good for trending markets, quick signals
   - **Weaknesses**: Whipsaws in choppy markets

4. **MeanReversionStrategy** (Weight: 0.1)
   - **Why**: Profits from temporary mispricings
   - **Strengths**: Good for range-bound markets
   - **Weaknesses**: Can lose in strong trends

### Ensemble Benefits

- **Diversification**: Different strategies capture different market regimes
- **Robustness**: Reduces reliance on any single approach
- **Consensus**: Requires majority agreement (≥2 strategies), reducing false signals
- **Weighted Voting**: More weight to proven strategies (ML gets 0.4)

## How Are Weights Chosen?

Current weights are **fixed** based on:
1. Historical performance (ML typically best)
2. Strategy reliability (Rule-based is stable)
3. Market coverage (Momentum for trends, Mean Reversion for ranges)

**Future Enhancement**: Meta-learning to optimize weights via cross-validation (see `src/pm_agent/strategies/meta_learning.py`)

## Strategy Selection Logic

```python
# Consensus required
if num_long >= 2 and weighted_strength > 0.5:
    return Signal(side='LONG', strength=weighted_strength)
```

This means:
- At least 2 strategies must agree on direction
- Weighted strength must exceed 0.5 (moderate confidence)
- Prevents single-strategy false signals

