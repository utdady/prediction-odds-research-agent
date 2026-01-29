# Market Intelligence Assistant - User Guide

## Overview

The Market Intelligence Assistant is an AI-powered agent that analyzes stock performance based on prediction market odds. It provides natural language insights about which stocks are performing well, which are struggling, and generates actionable recommendations.

## Features

### 1. **Daily Digest** 📰
Get a comprehensive daily summary of market activity, including:
- Market mood (Bullish/Bearish/Mixed)
- Top performers and decliners
- Sector performance
- Actionable insights

**Use Case**: Start your day with a quick overview of market sentiment.

### 2. **Stock Performance Analysis** 📈
Analyze all stocks over a customizable time period:
- Identify top performers
- Find stocks under pressure
- View neutral/stable stocks
- Customize lookback period and significance threshold

**Use Case**: Weekly review of portfolio holdings or finding new opportunities.

### 3. **Individual Stock Outlook** 🔍
Get detailed analysis for any stock:
- **Outlook**: Bullish, Bearish, or Neutral
- **Recommendation**: Buy/Sell/Hold with confidence level
- **Risk Level**: High, Moderate, or Low volatility
- **Key Metrics**: Probability, change, trend, liquidity
- **Recent Signals**: Latest trading signals generated

**Use Case**: Deep dive before making investment decisions.

### 4. **Stock Comparison** ⚖️
Compare 2-5 stocks side-by-side:
- Visual comparison of performance
- Confidence levels
- Risk assessment
- Recommendations ranked by outlook

**Use Case**: Choosing between multiple investment options.

### 5. **Sector Analysis** 🏭
Analyze performance by industry sector:
- Sector-level sentiment
- Average changes
- Number of stocks per sector
- Visual performance charts

**Use Case**: Understanding broader market trends and rotation.

## How It Works

### Data Sources
The assistant analyzes:
1. **Probability Changes** (`delta_p_1d`, `delta_p_1h`) - How odds are moving
2. **Current Probabilities** (`p_now`) - Market confidence levels
3. **Liquidity Scores** - Reliability of the data
4. **Volatility** (`rolling_std_p_1d`) - Risk indicators
5. **Trading Signals** - Generated from ensemble strategies

### Analysis Logic

#### Performance Classification:
- **Performing Well**: 
  - Average daily change > 8% (configurable threshold)
  - High liquidity (>30%)
  - OR total change > 16% over period
  
- **Performing Poorly**:
  - Average daily change < -8%
  - High liquidity
  - OR total change < -16% over period
  
- **Neutral**:
  - Changes within threshold
  - Stable probabilities

#### Confidence Scoring:
Confidence is based on **liquidity score**:
- **High Confidence** (>70%): High liquidity, reliable data
- **Moderate Confidence** (40-70%): Moderate liquidity
- **Low Confidence** (<40%): Low liquidity, less reliable

#### Risk Assessment:
Based on **volatility** (rolling standard deviation):
- **High Risk**: volatility > 0.10
- **Moderate Risk**: volatility > 0.05
- **Low Risk**: volatility ≤ 0.05

### Recommendations

The assistant generates recommendations by combining:
1. **Outlook** (Bullish/Bearish/Neutral)
2. **Confidence** (Liquidity score)
3. **Risk Level** (Volatility)

Examples:
- "Strong Buy - High confidence in upward momentum" (Bullish + High Confidence)
- "Watch - Positive signals but low liquidity" (Bullish + Low Confidence)
- "Strong Sell - High confidence in downward pressure" (Bearish + High Confidence)
- "Hold - No significant movement detected" (Neutral)

## Usage Examples

### Example 1: Morning Routine
```
1. Open "Market Intelligence" tab
2. Select "Daily Digest"
3. Click "Generate Today's Digest"
4. Review top movers, sector performance, and insights
5. Use sidebar "Quick Insights" for instant market mood
```

### Example 2: Researching a Stock
```
1. Select "Individual Stock"
2. Choose ticker (e.g., AAPL)
3. Click "Get Stock Outlook"
4. Review:
   - Outlook and recommendation
   - Confidence and risk level
   - Recent signals
   - Key metrics
```

### Example 3: Portfolio Review
```
1. Select "Stock Performance"
2. Set lookback period to 7 days
3. Click "Analyze All Stocks"
4. Review "Top Performers" table
5. Check "Under Pressure" stocks in your portfolio
```

### Example 4: Comparing Options
```
1. Select "Compare Stocks"
2. Choose 3-5 tickers you're considering
3. Click "Compare Stocks"
4. Review comparison table and chart
5. Identify best risk/reward opportunity
```

### Example 5: Sector Rotation
```
1. Select "Sector Analysis"
2. Click "Analyze Sectors"
3. Identify strongest and weakest sectors
4. Look for rotation opportunities
5. Use visualization to spot trends
```

## Integration with Existing Features

### With Trading Signals
The assistant displays recent trading signals for each stock, showing:
- Strategy used (Ensemble, ModelV1, etc.)
- Signal strength
- Side (LONG/SHORT)
- Horizon (days)

### With Backtest Results
Use the assistant to understand *why* certain signals were generated:
1. Check backtest results in "Backtest" tab
2. Use Market Intelligence to analyze current market conditions
3. Understand if signals align with broader trends

### With Search Agent
Combine with the conversational agent:
1. Ask search agent: "show signals for AAPL"
2. Use Market Intelligence for: "Why is AAPL generating signals?"
3. Get comprehensive view of stock

## Quick Insights Sidebar

Three instant analysis buttons:

1. **🚀 Best Stock Today**
   - Instant top performer
   - Shows change and reason
   
2. **📉 Worst Stock Today**
   - Instant worst performer
   - Shows decline and reason
   
3. **📊 Market Mood**
   - Overall sentiment
   - Percentage up/down
   - Bullish/Bearish/Mixed indicator

## Tips for Best Results

### 1. **Check Liquidity**
Always pay attention to the "liquidity note":
- ✅ "High liquidity - reliable" → Trust the analysis
- ⚠️ "Low liquidity - less reliable" → Be cautious

### 2. **Consider Time Horizon**
- **1-day lookback**: Intraday momentum, short-term trades
- **7-day lookback**: Weekly trends, swing trades
- **30-day lookback**: Monthly trends, position trades

### 3. **Risk Management**
- High volatility stocks = Higher risk
- Low liquidity + High volatility = Very high risk
- Consider diversification across sectors

### 4. **Combine Multiple Views**
Don't rely on a single metric:
1. Check individual stock outlook
2. Review sector performance
3. Compare with similar stocks
4. Verify with trading signals

### 5. **Use Daily Digest**
Make it part of your routine:
- Morning: Daily digest for market overview
- Mid-day: Individual stock analysis
- Evening: Sector analysis and portfolio review

## Limitations

### Data Limitations
- Based on prediction market odds only
- Requires sufficient historical data (7+ days)
- Low liquidity markets may give unreliable signals

### Analysis Limitations
- Not financial advice
- Past performance doesn't guarantee future results
- Should be combined with other analysis methods

### Technical Limitations
- Requires database with recent features
- Performance depends on data quality
- May be slow with large datasets

## Troubleshooting

### "No data available"
- Check if pipelines have run recently
- Verify markets and ticks are ingested
- Ensure features are built

### "Low liquidity" warnings
- Normal for newer or smaller markets
- Consider only high-liquidity stocks
- Use longer time periods for more data

### Inconsistent recommendations
- Markets are dynamic - updates happen
- Check timestamp of last update
- Verify with multiple analysis types

## API Access

For programmatic access, use the `StockAnalysisAgent` class directly:

```python
from stock_analysis_agent import StockAnalysisAgent
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg://pm:pm@localhost:5432/pm_research")
analyst = StockAnalysisAgent(engine)

# Get performance analysis
results = analyst.analyze_stock_performance(lookback_days=7)

# Get stock outlook
outlook = analyst.get_stock_outlook("AAPL")

# Compare stocks
comparison = analyst.compare_stocks(["AAPL", "MSFT", "GOOGL"])

# Get sector analysis
sectors = analyst.get_sector_analysis()

# Generate daily digest
digest = analyst.get_daily_digest()
print(digest)
```

## Future Enhancements

Potential improvements:
- ✨ Real-time alerts for significant changes
- ✨ Historical trend charts
- ✨ Export reports to PDF
- ✨ Custom watchlists
- ✨ Email/Slack notifications for daily digest
- ✨ Machine learning for sentiment scoring
- ✨ Integration with news sentiment
- ✨ Options market data integration

## Feedback

The Market Intelligence Assistant is continuously improving. Suggestions welcome!

---

**Remember**: This tool analyzes prediction market odds to provide insights. Always do your own research and consult with financial professionals before making investment decisions.


