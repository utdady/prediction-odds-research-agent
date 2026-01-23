"""Strategy comparison analysis (can be converted to notebook)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# Connect to database
engine = create_engine("postgresql+psycopg://pm:pm@localhost:5432/pm_research")

# Load signals by strategy
signals = pd.read_sql("""
    SELECT strategy, side, strength, ts, entity_id
    FROM signals
    ORDER BY ts
""", engine)

# Strategy performance comparison
strategy_stats = signals.groupby("strategy").agg({
    "strength": ["mean", "std", "count"],
    "entity_id": "nunique",
}).round(4)

print("Strategy Statistics:")
print(strategy_stats)

# Strategy correlation (if multiple strategies for same entity/time)
strategy_pivot = signals.pivot_table(
    index=["entity_id", "ts"],
    columns="strategy",
    values="strength",
    aggfunc="mean",
)

if len(strategy_pivot) > 0:
    corr = strategy_pivot.corr()
    fig = px.imshow(corr, text_auto=True, title="Strategy Correlation")
    fig.show()

