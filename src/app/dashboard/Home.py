from __future__ import annotations

import pandas as pd
import streamlit as st

import sqlalchemy

st.set_page_config(page_title="PM Odds Agent", layout="wide")

st.title("Prediction Market Odds → Equity Signals")

st.sidebar.header("DB")
db_url = st.sidebar.text_input("DATABASE_URL_SYNC", value="postgresql+psycopg://pm:pm@localhost:5432/pm_research")

try:
    engine = sqlalchemy.create_engine(db_url)

    tab1, tab2, tab3, tab4 = st.tabs(["Markets", "Signals", "Backtest", "Diagnostics"])

    with tab1:
        st.subheader("Latest markets")
        df = pd.read_sql("SELECT market_id, venue_id, title, status, updated_at FROM markets ORDER BY updated_at DESC", engine)
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("Latest signals")
        df = pd.read_sql("SELECT signal_id, entity_id, ts, strategy, side, strength, horizon_days FROM signals ORDER BY ts DESC", engine)
        st.dataframe(df, use_container_width=True)

    with tab3:
        st.subheader("Backtest runs")
        runs = pd.read_sql("SELECT r.run_id, r.created_at, m.sharpe, m.max_drawdown, m.cagr FROM backtest_runs r LEFT JOIN backtest_metrics m ON r.run_id=m.run_id ORDER BY r.created_at DESC", engine)
        st.dataframe(runs, use_container_width=True)

        if len(runs):
            run_id = st.selectbox("Select run", runs["run_id"].tolist())
            trades = pd.read_sql(f"SELECT * FROM backtest_trades WHERE run_id='{run_id}' ORDER BY entry_ts", engine)
            st.subheader("Trades")
            st.dataframe(trades, use_container_width=True)

            import glob
            import os
            import plotly.express as px

            paths = glob.glob(f"artifacts/equity_curve_{run_id}.csv")
            if paths:
                curve = pd.read_csv(paths[0], parse_dates=["date"]).sort_values("date")
                fig = px.line(curve, x="date", y="equity", title="Equity curve")
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Diagnostics")
        st.write("Calibration/Brier is logged during training; API exposes a minimal view.")
except Exception as e:
    st.error(f"Error connecting to database: {e}")

