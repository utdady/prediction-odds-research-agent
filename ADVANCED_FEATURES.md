# Advanced Features Implementation

This document describes the advanced features that have been implemented.

## ✅ Implemented Advanced Features

### 1. Ensemble of Multiple Strategies
**Location**: `src/pm_agent/strategies/ensemble.py`

- Combines 4 different strategies:
  - **RuleStrategyV1**: Simple rule-based (delta_p threshold)
  - **ModelStrategyV1**: ML model-based predictions
  - **MomentumStrategy**: Momentum-based signals
  - **MeanReversionStrategy**: Mean reversion signals
- Uses weighted voting (configurable weights)
- Requires consensus (majority of strategies agree)
- Integrated into `pipelines/run_inference.py` as `run_ensemble_strategy()`

**Usage**:
```python
from pm_agent.strategies.ensemble import EnsembleStrategy

ensemble = EnsembleStrategy()
signal = ensemble.generate_signal(features_dict)
```

### 2. Monte Carlo Simulation for Risk Analysis
**Location**: `src/pm_agent/risk/monte_carlo.py`

- Simulates thousands of possible future paths
- Calculates risk metrics:
  - Value at Risk (95% and 99%)
  - Expected Shortfall
  - Probability of profit/loss
  - Mean, median, min, max returns
- Available in dashboard (Backtest tab → "Run Monte Carlo Simulation")

**Usage**:
```python
from pm_agent.risk.monte_carlo import run_monte_carlo_simulation

results = run_monte_carlo_simulation(signals_df, config, n_simulations=10000)
print(f"95% VaR: {results['var_95']:.2%}")
print(f"Prob Profit: {results['prob_profit']:.1%}")
```

### 3. Cross-Venue Arbitrage Detection
**Location**: `src/pm_agent/arbitrage/detector.py` and `pipelines/detect_arbitrage.py`

- Detects mispricing between Kalshi and Polymarket
- Configurable minimum spread threshold (default 3%)
- Returns arbitrage opportunities with expected profit
- Available in dashboard (Advanced tab → "Scan for Arbitrage Opportunities")

**Usage**:
```python
from pm_agent.arbitrage.detector import ArbitrageDetector

detector = ArbitrageDetector(min_spread=0.03)
opportunities = detector.find_arbitrage_opportunities(markets_list)
```

### 4. Market Regime Detection
**Location**: `src/pm_agent/features/regime.py`

- Classifies market state: bull, bear, or choppy
- Uses Gaussian Mixture Model on SPY returns
- Provides adaptive thresholds based on regime
- Available in dashboard (Advanced tab → "Detect Current Regime")

**Usage**:
```python
from pm_agent.features.regime import detect_market_regime, get_regime_adaptive_threshold

regime = detect_market_regime(spy_returns)
threshold = get_regime_adaptive_threshold(regime, base_threshold=0.08)
```

### 5. Sentiment Analysis from News
**Location**: `src/pm_agent/features/sentiment.py`

- Fetches news headlines from Google News RSS
- Analyzes sentiment (keyword-based or FinBERT if available)
- Returns sentiment score (-1.0 to +1.0)
- Available in dashboard (Advanced tab → "Analyze Sentiment")

**Usage**:
```python
from pm_agent.features.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer(use_finbert=False)
sentiment = await analyzer.get_news_sentiment("AAPL", lookback_hours=24)
```

**Note**: For advanced FinBERT analysis, install:
```bash
pip install transformers torch
```

### 6. Tear Sheet PDF Generation
**Location**: `src/pm_agent/reports/tear_sheet.py`

- Generates professional PDF reports for backtest results
- Includes metrics, trade summary, and equity curve
- Falls back to text format if reportlab not installed
- Available in dashboard (Backtest tab → "Generate PDF Tear Sheet")

**Usage**:
```python
from pm_agent.reports.tear_sheet import generate_tear_sheet

pdf_path = generate_tear_sheet(
    run_id="abc123",
    metrics={"sharpe": 1.5, "max_drawdown": -0.10},
    trades_df=trades_df,
    equity_curve_df=curve_df,
)
```

## 📊 Dashboard Integration

All advanced features are accessible through the Streamlit dashboard:

1. **Backtest Tab**:
   - Monte Carlo Simulation button
   - Generate Tear Sheet button

2. **Advanced Tab** (new):
   - Market Regime Detection
   - Arbitrage Opportunity Scanner
   - News Sentiment Analysis

## 🔧 Configuration

No additional configuration required. All features work with existing settings.

Optional dependencies:
- `reportlab` - For PDF tear sheets (already in requirements.txt)
- `transformers` + `torch` - For FinBERT sentiment analysis (commented out in requirements.txt due to size)

## 🚀 Pipeline Integration

### Ensemble Strategy
Automatically runs in `pipelines/run_inference.py`:
```python
n_ensemble = await run_ensemble_strategy(session)
```

### Arbitrage Detection
Run separately:
```bash
python -m pipelines.detect_arbitrage
```

## 📈 Performance Notes

- **Monte Carlo**: Default 10,000 simulations (adjustable)
- **Regime Detection**: Requires >50 days of SPY data
- **Sentiment Analysis**: Fetches up to 10 recent articles
- **Ensemble**: Runs all 4 strategies in sequence

## 🔮 Future Enhancements

Not yet implemented but documented:
- Reinforcement Learning Agent (requires stable-baselines3)
- Live Paper Trading (requires real API integrations)
- Options-Implied Probability (requires options data source)
- Portfolio Optimization (Modern Portfolio Theory)
