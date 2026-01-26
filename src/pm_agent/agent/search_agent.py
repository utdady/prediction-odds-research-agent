"""Interactive search agent for querying company data, visualizations, and track records."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
import structlog

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

log = structlog.get_logger(__name__)


class SearchAgent:
    """Interactive agent that answers questions about companies, signals, and backtests."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def search(self, query: str) -> dict:
        """
        Process a user query and return relevant information.
        
        Args:
            query: User's question/query
        
        Returns:
            dict with 'answer', 'data', 'visualization', 'type'
        """
        query_lower = query.lower().strip()

        # Company information queries
        if any(word in query_lower for word in ["company", "ticker", "stock", "entity"]):
            return self._handle_company_query(query)

        # Signal queries
        if any(word in query_lower for word in ["signal", "signals", "recommendation", "buy", "sell"]):
            return self._handle_signal_query(query)

        # Backtest queries
        if any(word in query_lower for word in ["backtest", "performance", "returns", "sharpe", "drawdown"]):
            return self._handle_backtest_query(query)

        # Market queries
        if any(word in query_lower for word in ["market", "markets", "probability", "odds"]):
            return self._handle_market_query(query)

        # Feature queries
        if any(word in query_lower for word in ["feature", "features", "delta", "liquidity"]):
            return self._handle_feature_query(query)

        # General help
        if any(word in query_lower for word in ["help", "what can", "show me", "list"]):
            return self._handle_help_query()

        # Default: try to extract ticker and show company info
        ticker = self._extract_ticker(query)
        if ticker:
            return self._handle_company_query(f"show {ticker}")

        return {
            "type": "text",
            "answer": "I didn't understand your query. Try asking about:\n- Company information (e.g., 'show AAPL')\n- Signals (e.g., 'what signals do we have?')\n- Backtest performance (e.g., 'show backtest results')\n- Markets (e.g., 'show markets for AAPL')\n\nType 'help' for more options.",
            "data": None,
        }

    def _extract_ticker(self, query: str) -> str | None:
        """Extract ticker symbol from query."""
        # Common tickers
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX", "SPY"]
        query_upper = query.upper()
        for ticker in tickers:
            if ticker in query_upper:
                return ticker
        return None

    def _handle_company_query(self, query: str) -> dict:
        """Handle queries about companies."""
        ticker = self._extract_ticker(query)
        if not ticker:
            # List all companies
            df = pd.read_sql(
                """
                SELECT DISTINCT e.entity_id, e.ticker, e.name, e.sector
                FROM entities e
                ORDER BY e.ticker
                """,
                self.engine,
            )
            if len(df) == 0:
                return {
                    "type": "text",
                    "answer": "No companies found in database.",
                    "data": None,
                }
            return {
                "type": "table",
                "answer": f"Found {len(df)} companies in database:",
                "data": df,
            }

        # Get company details
        company_df = pd.read_sql(
            f"""
            SELECT e.entity_id, e.ticker, e.name, e.sector, e.created_at
            FROM entities e
            WHERE e.ticker = '{ticker}'
            """,
            self.engine,
        )

        if len(company_df) == 0:
            return {
                "type": "text",
                "answer": f"Company {ticker} not found in database.",
                "data": None,
            }

        entity_id = company_df.iloc[0]["entity_id"]

        # Get related markets
        markets_df = pd.read_sql(
            f"""
            SELECT m.market_id, m.title, m.status, m.venue_id, m.updated_at
            FROM markets m
            INNER JOIN market_entity_map mem ON m.market_id = mem.market_id
            WHERE mem.entity_id = '{entity_id}'
            ORDER BY m.updated_at DESC
            LIMIT 10
            """,
            self.engine,
        )

        # Get latest signals
        signals_df = pd.read_sql(
            f"""
            SELECT s.signal_id, s.ts, s.strategy, s.side, s.strength, s.horizon_days
            FROM signals s
            WHERE s.entity_id = '{entity_id}'
            ORDER BY s.ts DESC
            LIMIT 10
            """,
            self.engine,
        )

        # Get latest features
        features_df = pd.read_sql(
            f"""
            SELECT f.ts, f.p_now, f.delta_p_1d, f.liquidity_score, f.venue_disagreement
            FROM features f
            WHERE f.entity_id = '{entity_id}'
            ORDER BY f.ts DESC
            LIMIT 10
            """,
            self.engine,
        )

        answer_parts = [f"**{ticker}** ({company_df.iloc[0]['name']})"]
        if company_df.iloc[0]["sector"]:
            answer_parts.append(f"Sector: {company_df.iloc[0]['sector']}")

        answer_parts.append(f"\n**Markets:** {len(markets_df)} found")
        answer_parts.append(f"**Signals:** {len(signals_df)} found")
        answer_parts.append(f"**Features:** {len(features_df)} found")

        return {
            "type": "company_detail",
            "answer": "\n".join(answer_parts),
            "data": {
                "company": company_df,
                "markets": markets_df,
                "signals": signals_df,
                "features": features_df,
            },
        }

    def _handle_signal_query(self, query: str) -> dict:
        """Handle queries about signals."""
        ticker = self._extract_ticker(query)

        if ticker:
            sql = f"""
                SELECT s.entity_id, s.ts, s.strategy, s.side, s.strength, s.horizon_days
                FROM signals s
                INNER JOIN entities e ON s.entity_id = e.entity_id
                WHERE e.ticker = '{ticker}'
                ORDER BY s.ts DESC
                LIMIT 20
            """
            answer = f"Signals for {ticker}:"
        else:
            sql = """
                SELECT e.ticker, s.ts, s.strategy, s.side, s.strength, s.horizon_days
                FROM signals s
                INNER JOIN entities e ON s.entity_id = e.entity_id
                ORDER BY s.ts DESC
                LIMIT 20
            """
            answer = "Recent signals:"

        df = pd.read_sql(sql, self.engine)

        if len(df) == 0:
            return {
                "type": "text",
                "answer": "No signals found.",
                "data": None,
            }

        return {
            "type": "table",
            "answer": answer,
            "data": df,
        }

    def _handle_backtest_query(self, query: str) -> dict:
        """Handle queries about backtest performance."""
        # Get latest backtest runs
        runs_df = pd.read_sql(
            """
            SELECT r.run_id, r.created_at, r.notes,
                   m.sharpe, m.max_drawdown, m.cagr, m.total_return
            FROM backtest_runs r
            LEFT JOIN backtest_metrics m ON r.run_id = m.run_id
            ORDER BY r.created_at DESC
            LIMIT 10
            """,
            self.engine,
        )

        if len(runs_df) == 0:
            return {
                "type": "text",
                "answer": "No backtest results found.",
                "data": None,
            }

        # Calculate summary stats
        if "sharpe" in runs_df.columns and runs_df["sharpe"].notna().any():
            avg_sharpe = runs_df["sharpe"].mean()
            best_sharpe = runs_df["sharpe"].max()
            answer = f"**Backtest Performance Summary:**\n"
            answer += f"- Average Sharpe Ratio: {avg_sharpe:.2f}\n"
            answer += f"- Best Sharpe Ratio: {best_sharpe:.2f}\n"
            answer += f"- Total Runs: {len(runs_df)}\n"
        else:
            answer = f"Found {len(runs_df)} backtest runs."

        return {
            "type": "table",
            "answer": answer,
            "data": runs_df,
        }

    def _handle_market_query(self, query: str) -> dict:
        """Handle queries about markets."""
        ticker = self._extract_ticker(query)

        if ticker:
            sql = f"""
                SELECT m.market_id, m.title, m.status, m.venue_id, m.updated_at,
                       (SELECT p_norm FROM odds_ticks 
                        WHERE market_id = m.market_id 
                        ORDER BY tick_ts DESC LIMIT 1) as probability
                FROM markets m
                INNER JOIN market_entity_map mem ON m.market_id = mem.market_id
                INNER JOIN entities e ON mem.entity_id = e.entity_id
                WHERE e.ticker = '{ticker}'
                ORDER BY m.updated_at DESC
                LIMIT 20
            """
            answer = f"Markets for {ticker}:"
        else:
            sql = """
                SELECT m.market_id, m.title, m.status, m.venue_id, m.updated_at,
                       (SELECT p_norm FROM odds_ticks 
                        WHERE market_id = m.market_id 
                        ORDER BY tick_ts DESC LIMIT 1) as probability
                FROM markets m
                ORDER BY m.updated_at DESC
                LIMIT 20
            """
            answer = "Recent markets:"

        df = pd.read_sql(sql, self.engine)

        if len(df) == 0:
            return {
                "type": "text",
                "answer": "No markets found.",
                "data": None,
            }

        return {
            "type": "table",
            "answer": answer,
            "data": df,
        }

    def _handle_feature_query(self, query: str) -> dict:
        """Handle queries about features."""
        ticker = self._extract_ticker(query)

        if ticker:
            sql = f"""
                SELECT f.ts, f.p_now, f.delta_p_1d, f.liquidity_score, 
                       f.venue_disagreement, f.rolling_std_p_1d
                FROM features f
                INNER JOIN entities e ON f.entity_id = e.entity_id
                WHERE e.ticker = '{ticker}'
                ORDER BY f.ts DESC
                LIMIT 20
            """
            answer = f"Features for {ticker}:"
        else:
            sql = """
                SELECT e.ticker, f.ts, f.p_now, f.delta_p_1d, f.liquidity_score
                FROM features f
                INNER JOIN entities e ON f.entity_id = e.entity_id
                ORDER BY f.ts DESC
                LIMIT 20
            """
            answer = "Recent features:"

        df = pd.read_sql(sql, self.engine)

        if len(df) == 0:
            return {
                "type": "text",
                "answer": "No features found.",
                "data": None,
            }

        return {
            "type": "table",
            "answer": answer,
            "data": df,
        }

    def _handle_help_query(self) -> dict:
        """Handle help queries."""
        help_text = """
**I can help you find information about:**

1. **Companies** - Ask about any ticker
   - "show AAPL"
   - "what companies do we have?"
   - "company information for MSFT"

2. **Signals** - Trading signals and recommendations
   - "show signals for AAPL"
   - "what signals do we have?"
   - "recent buy signals"

3. **Backtest Performance** - Strategy performance
   - "show backtest results"
   - "what's our Sharpe ratio?"
   - "backtest performance"

4. **Markets** - Prediction markets
   - "markets for AAPL"
   - "show all markets"
   - "what markets do we track?"

5. **Features** - Market features and indicators
   - "features for AAPL"
   - "show feature data"
   - "delta_p_1d for MSFT"

**Examples:**
- "show AAPL" - Get all info about Apple
- "signals for TSLA" - Get trading signals for Tesla
- "backtest results" - See performance metrics
- "markets" - List all markets
        """
        return {
            "type": "text",
            "answer": help_text,
            "data": None,
        }

