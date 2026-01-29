"""Enhanced stock analysis agent that interprets prediction market odds."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import structlog
from sqlalchemy.engine import Engine

log = structlog.get_logger(__name__)


class StockAnalysisAgent:
    """Intelligent agent that analyzes stocks based on prediction market odds."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------ #
    # Core performance analysis
    # ------------------------------------------------------------------ #
    def analyze_stock_performance(
        self,
        ticker: str | None = None,
        lookback_days: int = 7,
        threshold: float = 0.08,
    ) -> dict[str, Any]:
        """
        Analyze which stocks are doing well/poorly based on odds changes.

        Args:
            ticker: Specific ticker to analyze (None = all tickers)
            lookback_days: Days to look back for analysis
            threshold: Delta threshold for "significant" movement

        Returns:
            dict with 'performing_well', 'performing_poorly', 'neutral', 'analysis'
        """
        # Use latest feature timestamp as reference so mock/historic data still works
        ref_query = "SELECT MAX(ts) AS max_ts FROM features"
        ref_df = pd.read_sql(ref_query, self.engine)
        ref_ts = ref_df["max_ts"].iloc[0]
        if pd.isna(ref_ts):
            return {
                "performing_well": [],
                "performing_poorly": [],
                "neutral": [],
                "analysis": "No feature data available for analysis.",
            }

        # Treat reference as \"now\" and look back from there
        reference_time = pd.to_datetime(ref_ts).to_pydatetime()
        cutoff = reference_time - timedelta(days=lookback_days)

        # Use parameterized queries to prevent SQL injection
        if ticker:
            query = """
                SELECT 
                    e.ticker,
                    e.name,
                    e.sector,
                    f.ts,
                    f.p_now,
                    f.delta_p_1d,
                    f.delta_p_1h,
                    f.liquidity_score,
                    f.venue_disagreement,
                    f.rolling_std_p_1d
                FROM features f
                JOIN entities e ON f.entity_id = e.entity_id
                WHERE e.ticker = %s AND f.ts >= %s
                ORDER BY f.ts DESC
            """
            params = (ticker, cutoff.isoformat())
        else:
            query = """
                SELECT 
                    e.ticker,
                    e.name,
                    e.sector,
                    f.ts,
                    f.p_now,
                    f.delta_p_1d,
                    f.delta_p_1h,
                    f.liquidity_score,
                    f.venue_disagreement,
                    f.rolling_std_p_1d
                FROM features f
                JOIN entities e ON f.entity_id = e.entity_id
                WHERE f.ts >= %s
                ORDER BY e.ticker, f.ts DESC
            """
            params = (cutoff.isoformat(),)

        df = pd.read_sql(query, self.engine, params=params)

        if len(df) == 0:
            return {
                "performing_well": [],
                "performing_poorly": [],
                "neutral": [],
                "analysis": "No recent data available for analysis.",
            }

        analysis_by_ticker: dict[str, dict[str, Any]] = {}

        for ticker_name, group in df.groupby("ticker"):
            group = group.sort_values("ts")
            if len(group) < 2:
                continue

            latest = group.iloc[-1]
            oldest = group.iloc[0]

            avg_delta_1d = group["delta_p_1d"].mean()
            avg_p_now = group["p_now"].mean()
            total_change = latest["p_now"] - oldest["p_now"]
            volatility = group["delta_p_1d"].std()
            avg_liquidity = group["liquidity_score"].mean()
            trend_direction = (
                "up" if total_change > 0 else "down" if total_change < 0 else "flat"
            )

            # Sentiment classification
            if avg_delta_1d > threshold and avg_liquidity > 0.3:
                sentiment = "performing_well"
                reason = f"Strong upward momentum ({avg_delta_1d:.2%} avg daily change)"
            elif avg_delta_1d < -threshold and avg_liquidity > 0.3:
                sentiment = "performing_poorly"
                reason = f"Downward pressure ({avg_delta_1d:.2%} avg daily change)"
            elif abs(total_change) > threshold * 2:
                if total_change > 0:
                    sentiment = "performing_well"
                    reason = f"Significant probability increase ({total_change:.2%} over period)"
                else:
                    sentiment = "performing_poorly"
                    reason = f"Significant probability decrease ({total_change:.2%} over period)"
            else:
                sentiment = "neutral"
                reason = "Stable probabilities, no significant movement"

            # Risk assessment
            if volatility > 0.1:
                risk = "High volatility - uncertain"
            elif volatility > 0.05:
                risk = "Moderate volatility"
            else:
                risk = "Low volatility - stable"

            # Liquidity assessment
            if avg_liquidity < 0.2:
                liquidity_note = "Low liquidity - less reliable"
            elif avg_liquidity < 0.5:
                liquidity_note = "Moderate liquidity"
            else:
                liquidity_note = "High liquidity - reliable"

            analysis_by_ticker[ticker_name] = {
                "ticker": ticker_name,
                "name": latest["name"],
                "sector": latest["sector"],
                "sentiment": sentiment,
                "current_probability": float(latest["p_now"]),
                "total_change": float(total_change),
                "avg_daily_change": float(avg_delta_1d),
                "volatility": float(volatility) if pd.notna(volatility) else 0.0,
                "liquidity": float(avg_liquidity),
                "trend_direction": trend_direction,
                "reason": reason,
                "risk": risk,
                "liquidity_note": liquidity_note,
                "last_updated": latest["ts"].isoformat() if latest["ts"] else None,
            }

        # Categorize
        performing_well: list[dict[str, Any]] = []
        performing_poorly: list[dict[str, Any]] = []
        neutral: list[dict[str, Any]] = []

        for analysis in analysis_by_ticker.values():
            if analysis["sentiment"] == "performing_well":
                performing_well.append(analysis)
            elif analysis["sentiment"] == "performing_poorly":
                performing_poorly.append(analysis)
            else:
                neutral.append(analysis)

        performing_well.sort(key=lambda x: x["avg_daily_change"], reverse=True)
        performing_poorly.sort(key=lambda x: x["avg_daily_change"])

        summary = self._generate_summary(
            performing_well, performing_poorly, neutral, lookback_days
        )

        return {
            "performing_well": performing_well,
            "performing_poorly": performing_poorly,
            "neutral": neutral,
            "analysis": summary,
            "lookback_days": lookback_days,
            "threshold": threshold,
        }

    def _generate_summary(
        self,
        performing_well: list[dict[str, Any]],
        performing_poorly: list[dict[str, Any]],
        neutral: list[dict[str, Any]],
        lookback_days: int,
    ) -> str:
        lines = [f"## Stock Performance Analysis (Last {lookback_days} Days)", ""]

        if performing_well:
            lines.append("### Top Performers")
            for stock in performing_well[:3]:
                lines.append(
                    f"- **{stock['ticker']}** ({stock['name']}): "
                    f"{stock['current_probability']:.1%} probability "
                    f"({stock['total_change']:+.1%} change). "
                    f"{stock['reason']}. {stock['liquidity_note']}."
                )
            lines.append("")

        if performing_poorly:
            lines.append("### Stocks Under Pressure")
            for stock in performing_poorly[:3]:
                lines.append(
                    f"- **{stock['ticker']}** ({stock['name']}): "
                    f"{stock['current_probability']:.1%} probability "
                    f"({stock['total_change']:+.1%} change). "
                    f"{stock['reason']}. {stock['liquidity_note']}."
                )
            lines.append("")

        if neutral:
            lines.append(
                f"### Stable Stocks: {len(neutral)} stocks showing no significant movement"
            )
            lines.append("")

        total = len(performing_well) + len(performing_poorly) + len(neutral)
        if total > 0:
            pct_up = len(performing_well) / total * 100
            pct_down = len(performing_poorly) / total * 100

            lines.append("### Overall Market Sentiment")
            if pct_up > 60:
                lines.append(f"**Bullish**: {pct_up:.0f}% of stocks positive")
            elif pct_down > 60:
                lines.append(f"**Bearish**: {pct_down:.0f}% of stocks negative")
            else:
                lines.append(
                    f"**Mixed**: {pct_up:.0f}% up, {pct_down:.0f}% down, {100-pct_up-pct_down:.0f}% neutral"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Derived views
    # ------------------------------------------------------------------ #
    def get_stock_outlook(self, ticker: str) -> dict[str, Any]:
        """Get detailed outlook for a specific stock."""
        performance = self.analyze_stock_performance(ticker=ticker, lookback_days=7)

        if (
            not performance["performing_well"]
            and not performance["performing_poorly"]
            and not performance["neutral"]
        ):
            return {
                "ticker": ticker,
                "outlook": "unknown",
                "recommendation": "No data available",
                "confidence": 0.0,
                "details": {},
            }

        stock_data: dict[str, Any] | None = None
        outlook_category: str | None = None

        for stock in performance["performing_well"]:
            if stock["ticker"] == ticker:
                stock_data = stock
                outlook_category = "bullish"
                break

        if not stock_data:
            for stock in performance["performing_poorly"]:
                if stock["ticker"] == ticker:
                    stock_data = stock
                    outlook_category = "bearish"
                    break

        if not stock_data:
            for stock in performance["neutral"]:
                if stock["ticker"] == ticker:
                    stock_data = stock
                    outlook_category = "neutral"
                    break

        if not stock_data:
            return {
                "ticker": ticker,
                "outlook": "unknown",
                "recommendation": "Insufficient data",
                "confidence": 0.0,
                "details": {},
            }

        confidence = stock_data["liquidity"]

        if outlook_category == "bullish":
            if confidence > 0.7:
                recommendation = "Strong Buy - High confidence in upward momentum"
            elif confidence > 0.4:
                recommendation = "Buy - Positive momentum, moderate confidence"
            else:
                recommendation = "Watch - Positive signals but low liquidity"
        elif outlook_category == "bearish":
            if confidence > 0.7:
                recommendation = "Strong Sell - High confidence in downward pressure"
            elif confidence > 0.4:
                recommendation = "Sell - Negative momentum, moderate confidence"
            else:
                recommendation = "Watch - Negative signals but low liquidity"
        else:
            recommendation = "Hold - No significant movement detected"

        volatility = stock_data["volatility"]
        if volatility > 0.1:
            risk_level = "High"
        elif volatility > 0.05:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        signals = self._get_recent_signals(ticker)

        return {
            "ticker": ticker,
            "name": stock_data["name"],
            "sector": stock_data["sector"],
            "outlook": outlook_category,
            "recommendation": recommendation,
            "confidence": confidence,
            "risk_level": risk_level,
            "details": {
                "current_probability": stock_data["current_probability"],
                "total_change": stock_data["total_change"],
                "avg_daily_change": stock_data["avg_daily_change"],
                "volatility": volatility,
                "trend_direction": stock_data["trend_direction"],
                "reason": stock_data["reason"],
                "liquidity_note": stock_data["liquidity_note"],
            },
            "recent_signals": signals,
        }

    def _get_recent_signals(self, ticker: str) -> list[dict[str, Any]]:
        """Get recent trading signals for a ticker."""
        query = """
            SELECT s.ts, s.strategy, s.side, s.strength, s.horizon_days
            FROM signals s
            JOIN entities e ON s.entity_id = e.entity_id
            WHERE e.ticker = %s
            ORDER BY s.ts DESC
            LIMIT 5
        """

        try:
            df = pd.read_sql(query, self.engine, params=(ticker,))
            return df.to_dict("records") if len(df) > 0 else []
        except Exception:
            return []

    def get_sector_analysis(self) -> dict[str, Any]:
        """Analyze performance by sector."""
        # Use latest feature timestamp as reference
        ref_query = "SELECT MAX(ts) AS max_ts FROM features"
        ref_df = pd.read_sql(ref_query, self.engine)
        ref_ts = ref_df["max_ts"].iloc[0]
        if pd.isna(ref_ts):
            return {"sectors": [], "analysis": "No sector data available"}

        reference_time = pd.to_datetime(ref_ts).to_pydatetime()
        cutoff = reference_time - timedelta(days=7)

        query = """
            SELECT 
                e.sector,
                e.ticker,
                f.delta_p_1d,
                f.p_now,
                f.liquidity_score
            FROM features f
            JOIN entities e ON f.entity_id = e.entity_id
            WHERE f.ts >= %s
                AND e.sector IS NOT NULL
            ORDER BY e.sector, f.ts DESC
        """

        df = pd.read_sql(query, self.engine, params=(cutoff.isoformat(),))

        if len(df) == 0:
            return {"sectors": [], "analysis": "No sector data available"}

        sector_analysis: list[dict[str, Any]] = []

        for sector, group in df.groupby("sector"):
            avg_delta = group["delta_p_1d"].mean()
            avg_prob = group["p_now"].mean()
            n_stocks = group["ticker"].nunique()
            avg_liquidity = group["liquidity_score"].mean()

            if avg_delta > 0.05:
                sentiment, emoji = "strong", "🚀"
            elif avg_delta > 0.02:
                sentiment, emoji = "positive", "📈"
            elif avg_delta < -0.05:
                sentiment, emoji = "weak", "📉"
            elif avg_delta < -0.02:
                sentiment, emoji = "negative", "⚠️"
            else:
                sentiment, emoji = "neutral", "⚖️"

            sector_analysis.append(
                {
                    "sector": sector,
                    "sentiment": sentiment,
                    "avg_change": float(avg_delta),
                    "avg_probability": float(avg_prob),
                    "n_stocks": int(n_stocks),
                    "liquidity": float(avg_liquidity),
                    "emoji": emoji,
                }
            )

        sector_analysis.sort(key=lambda x: x["avg_change"], reverse=True)

        summary_lines = ["## Sector Performance", ""]
        for s in sector_analysis:
            summary_lines.append(
                f"{s['emoji']} **{s['sector']}**: "
                f"{s['sentiment'].title()} "
                f"({s['avg_change']:+.1%} avg change, {s['n_stocks']} stocks)"
            )

        return {"sectors": sector_analysis, "analysis": "\n".join(summary_lines)}

    def compare_stocks(self, tickers: list[str]) -> dict[str, Any]:
        """Compare multiple stocks side-by-side.

        Returns
        -------
        dict
            {
              "comparison": [outlook_dict, ...],
              "summary": markdown_summary,
            }
        """
        comparisons: list[dict[str, Any]] = []

        for ticker in tickers:
            outlook = self.get_stock_outlook(ticker)
            if outlook.get("outlook") != "unknown":
                comparisons.append(outlook)

        if not comparisons:
            return {"comparison": [], "summary": "No data available for comparison"}

        # Sort by outlook/bias and confidence
        def sort_key(x: dict[str, Any]) -> tuple[int, float]:
            outlook = x.get("outlook")
            if outlook == "bullish":
                bias = 2
            elif outlook == "neutral":
                bias = 1
            elif outlook == "bearish":
                bias = 0
            else:
                bias = -1
            return (bias, float(x.get("confidence", 0.0)))

        comparisons.sort(key=sort_key, reverse=True)

        best = comparisons[0]
        worst = comparisons[-1] if len(comparisons) > 1 else None

        summary_lines = ["## Stock Comparison", ""]
        summary_lines.append(
            f"**Best Outlook**: {best['ticker']} - {best['outlook'].title()} "
            f"(Confidence: {best['confidence']:.0%})"
        )
        if worst and worst is not best:
            summary_lines.append(
                f"**Weakest Outlook**: {worst['ticker']} - {worst['outlook'].title()} "
                f"(Confidence: {worst['confidence']:.0%})"
            )

        return {"comparison": comparisons, "summary": "\n".join(summary_lines)}

    def get_daily_digest(self) -> str:
        """Generate a daily digest of market activity (markdown)."""
        performance = self.analyze_stock_performance(lookback_days=1)
        sectors = self.get_sector_analysis()

        top_movers = performance["performing_well"][:3]
        bottom_movers = performance["performing_poorly"][:3]

        digest: list[str] = ["# Daily Market Digest"]
        digest.append(f"*{datetime.utcnow().strftime('%B %d, %Y')}*")
        digest.append("")

        digest.append("## Market Overview")
        total_stocks = (
            len(performance["performing_well"])
            + len(performance["performing_poorly"])
            + len(performance["neutral"])
        )

        if total_stocks > 0:
            pct_up = len(performance["performing_well"]) / total_stocks * 100
            pct_down = len(performance["performing_poorly"]) / total_stocks * 100

            if pct_up > 60:
                mood = "Bullish"
            elif pct_down > 60:
                mood = "Bearish"
            else:
                mood = "Mixed"

            digest.append(f"**Market Mood**: {mood}")
            digest.append(f"- {len(performance['performing_well'])} stocks up ({pct_up:.0f}%)")
            digest.append(f"- {len(performance['performing_poorly'])} stocks down ({pct_down:.0f}%)")
            digest.append(
                f"- {len(performance['neutral'])} stocks neutral "
                f"({100-pct_up-pct_down:.0f}%)"
            )
        digest.append("")

        if top_movers:
            digest.append("## Top Movers (24h)")
            for stock in top_movers:
                digest.append(
                    f"- **{stock['ticker']}**: {stock['total_change']:+.1%} - {stock['reason']}"
                )
            digest.append("")

        if bottom_movers:
            digest.append("## Biggest Declines (24h)")
            for stock in bottom_movers:
                digest.append(
                    f"- **{stock['ticker']}**: {stock['total_change']:+.1%} - {stock['reason']}"
                )
            digest.append("")

        digest.append(sectors["analysis"])
        digest.append("")

        digest.append("## Actionable Insights")
        if top_movers and top_movers[0]["liquidity"] > 0.6:
            digest.append(
                f"- Consider **{top_movers[0]['ticker']}** - strong upward momentum with high liquidity"
            )
        if bottom_movers and bottom_movers[0]["liquidity"] > 0.6:
            digest.append(
                f"- Monitor **{bottom_movers[0]['ticker']}** - significant downward pressure"
            )
        if len(performance["performing_well"]) == 0:
            digest.append("- No stocks showing strong positive signals today.")

        digest.append("")
        digest.append("---")
        digest.append(
            "*This digest is generated from prediction market odds data. Not financial advice.*"
        )

        return "\n".join(digest)


