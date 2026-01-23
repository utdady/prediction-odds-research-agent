"""Risk analysis (can be converted to notebook)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# Connect to database
engine = create_engine("postgresql+psycopg://pm:pm@localhost:5432/pm_research")

# Load backtest results
runs = pd.read_sql("""
    SELECT r.run_id, r.created_at, m.*
    FROM backtest_runs r
    LEFT JOIN backtest_metrics m ON r.run_id = m.run_id
    ORDER BY r.created_at DESC
""", engine)

if len(runs) > 0:
    # Risk metrics comparison
    risk_cols = ["sharpe", "max_drawdown", "calmar", "omega_ratio"]
    available_cols = [c for c in risk_cols if c in runs.columns]
    
    if available_cols:
        fig = px.scatter_matrix(
            runs[available_cols],
            title="Risk Metrics Comparison"
        )
        fig.show()
    
    # Drawdown analysis
    if "max_drawdown" in runs.columns:
        fig = px.histogram(
            runs,
            x="max_drawdown",
            title="Max Drawdown Distribution"
        )
        fig.show()

