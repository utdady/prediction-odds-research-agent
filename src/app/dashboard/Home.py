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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Markets", "Signals", "Backtest", "Diagnostics", "Advanced"])

    with tab1:
        st.subheader("Latest markets")
        df = pd.read_sql("SELECT market_id, venue_id, title, status, updated_at FROM markets ORDER BY updated_at DESC", engine)
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("Latest signals")
        df = pd.read_sql("SELECT signal_id, entity_id, ts, strategy, side, strength, horizon_days FROM signals ORDER BY ts DESC", engine)
        st.dataframe(df, use_container_width=True)
        
        # Signal Heatmap
        if len(df) > 0:
            st.subheader("Signal Heatmap")
            import plotly.express as px
            
            df_heatmap = pd.read_sql("""
                SELECT 
                    DATE(ts) as date,
                    entity_id,
                    strategy,
                    AVG(strength) as avg_strength
                FROM signals
                GROUP BY DATE(ts), entity_id, strategy
                ORDER BY date, entity_id
            """, engine)
            
            if len(df_heatmap) > 0:
                pivot = df_heatmap.pivot_table(
                    index='entity_id',
                    columns='date',
                    values='avg_strength',
                    aggfunc='mean'
                )
                
                if not pivot.empty:
                    fig = px.imshow(
                        pivot,
                        labels=dict(x="Date", y="Ticker", color="Signal Strength"),
                        color_continuous_scale="RdYlGn",
                        aspect="auto",
                        title="Signal Strength Heatmap"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for heatmap")
            else:
                st.info("No aggregated signal data available")

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
            
            # Monte Carlo Risk Analysis
            st.subheader("Monte Carlo Risk Analysis")
            if st.button("Run Monte Carlo Simulation"):
                try:
                    from pm_agent.risk.monte_carlo import run_monte_carlo_simulation
                    from pm_agent.backtest.engine import BacktestConfig, CostModel
                    from pm_agent.config import settings
                    
                    # Get signals for this run
                    signals_df = pd.read_sql(f"""
                        SELECT DISTINCT s.entity_id, s.ts, s.horizon_days
                        FROM signals s
                        JOIN backtest_trades bt ON s.entity_id = bt.entity_id
                        WHERE bt.run_id = '{run_id}'
                        ORDER BY s.ts
                    """, engine)
                    
                    if len(signals_df) > 0:
                        config = BacktestConfig(
                            max_positions=10,
                            holding_days=5,
                            cost_model=CostModel(settings.cost_spread_bps, settings.cost_slippage_bps),
                        )
                        
                        results = run_monte_carlo_simulation(signals_df, config, n_simulations=10000)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Probability of Profit", f"{results['prob_profit']:.1%}")
                        col2.metric("95% Value at Risk", f"{results['var_95']:.2%}")
                        col3.metric("Expected Shortfall", f"{results['expected_shortfall']:.2%}")
                        col4.metric("Mean Return", f"{results['mean_return']:.2%}")
                        
                        # Distribution plot
                        fig = px.histogram(
                            x=results['simulation_results'],
                            nbins=50,
                            title="Distribution of Simulated Returns",
                            labels={"x": "Final Equity", "y": "Frequency"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk metrics table
                        st.subheader("Risk Metrics")
                        risk_df = pd.DataFrame({
                            "Metric": ["Mean Return", "Std Return", "Median Return", "Min Return", "Max Return", 
                                      "95% VaR", "99% VaR", "Expected Shortfall", "Prob Profit", "Prob Loss"],
                            "Value": [
                                f"{results['mean_return']:.2%}",
                                f"{results['std_return']:.2%}",
                                f"{results['median_return']:.2%}",
                                f"{results['min_return']:.2%}",
                                f"{results['max_return']:.2%}",
                                f"{results['var_95']:.2%}",
                                f"{results['var_99']:.2%}",
                                f"{results['expected_shortfall']:.2%}",
                                f"{results['prob_profit']:.1%}",
                                f"{results['prob_loss']:.1%}",
                            ]
                        })
                        st.dataframe(risk_df, use_container_width=True)
                    else:
                        st.warning("No signals found for this run")
                except Exception as e:
                    st.error(f"Monte Carlo simulation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Tear Sheet Generation
            st.subheader("Generate Tear Sheet")
            if st.button("Generate PDF Tear Sheet"):
                try:
                    from pm_agent.reports.tear_sheet import generate_tear_sheet
                    
                    # Get metrics
                    metrics_row = runs[runs['run_id'] == run_id].iloc[0]
                    metrics = {
                        'sharpe': float(metrics_row.get('sharpe', 0.0)),
                        'max_drawdown': float(metrics_row.get('max_drawdown', 0.0)),
                        'cagr': float(metrics_row.get('cagr', 0.0)),
                    }
                    
                    # Calculate additional metrics
                    if len(trades) > 0:
                        winning = len(trades[trades.get('pnl_pct', 0) > 0])
                        metrics['win_rate'] = winning / len(trades)
                        metrics['total_return'] = trades['pnl_pct'].sum()
                    
                    # Load equity curve if available
                    equity_curve = None
                    if paths:
                        equity_curve = pd.read_csv(paths[0], parse_dates=["date"])
                    
                    pdf_path = generate_tear_sheet(
                        run_id=run_id,
                        output_path="artifacts",
                        metrics=metrics,
                        trades_df=trades,
                        equity_curve_df=equity_curve,
                    )
                    
                    st.success(f"Tear sheet generated: {pdf_path}")
                    st.info("Note: Install 'reportlab' for full PDF generation. Text version created otherwise.")
                except Exception as e:
                    st.error(f"Tear sheet generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Interactive Backtesting Tool
        st.subheader("Interactive Backtest")
        st.sidebar.header("Backtest Parameters")
        
        delta_threshold = st.sidebar.slider("Delta Threshold", 0.0, 0.3, 0.08, 0.01)
        min_liquidity = st.sidebar.slider("Min Liquidity", 0.0, 1.0, 0.2, 0.05)
        holding_days = st.sidebar.slider("Holding Period (days)", 1, 30, 5, 1)
        max_positions = st.sidebar.slider("Max Positions", 1, 20, 10, 1)
        use_exit_rules = st.sidebar.checkbox("Use Exit Rules", value=True)
        stop_loss = st.sidebar.slider("Stop Loss %", -0.20, 0.0, -0.05, 0.01)
        take_profit = st.sidebar.slider("Take Profit %", 0.0, 0.50, 0.10, 0.01)
        
        if st.button("Run Custom Backtest"):
            try:
                from pm_agent.backtest.engine import run_event_driven_backtest, BacktestConfig, CostModel, sharpe, max_drawdown
                from pm_agent.config import settings
                
                # Fetch signals based on custom parameters
                signals_df = pd.read_sql(f"""
                    SELECT DISTINCT s.entity_id, s.ts, s.horizon_days
                    FROM signals s
                    JOIN features f ON s.entity_id = f.entity_id AND DATE(s.ts) = DATE(f.ts)
                    WHERE f.delta_p_1d > {delta_threshold}
                      AND f.liquidity_score >= {min_liquidity}
                    ORDER BY s.ts
                """, engine)
                
                if len(signals_df) > 0:
                    config = BacktestConfig(
                        max_positions=max_positions,
                        holding_days=holding_days,
                        cost_model=CostModel(settings.cost_spread_bps, settings.cost_slippage_bps),
                        use_exit_rules=use_exit_rules,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit,
                    )
                    
                    curve, trades = run_event_driven_backtest(signals_df, config)
                    
                    if len(curve) > 0:
                        # Calculate metrics
                        curve['returns'] = curve['equity'].pct_change().fillna(0)
                        sharpe_ratio = sharpe(curve['returns'])
                        max_dd = max_drawdown(curve['equity'])
                        total_return = (curve['equity'].iloc[-1] / curve['equity'].iloc[0]) - 1.0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        col2.metric("Max Drawdown", f"{max_dd:.2%}")
                        col3.metric("Total Return", f"{total_return:.2%}")
                        col4.metric("Total Trades", len(trades))
                        
                        # Plot equity curve
                        fig = px.line(curve, x="date", y="equity", title="Custom Backtest Equity Curve")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show trades
                        if len(trades) > 0:
                            trades_df = pd.DataFrame([{
                                'entity_id': t.entity_id,
                                'entry_ts': t.entry_ts,
                                'exit_ts': t.exit_ts,
                                'entry_px': t.entry_px,
                                'exit_px': t.exit_px,
                                'pnl_pct': t.pnl_pct,
                            } for t in trades])
                            st.dataframe(trades_df, use_container_width=True)
                    else:
                        st.warning("No trades generated with these parameters")
                else:
                    st.warning("No signals match the criteria")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    with tab4:
        st.subheader("Diagnostics")
        st.write("Calibration/Brier is logged during training; API exposes a minimal view.")
        
        # Monte Carlo Risk Analysis
        st.subheader("Monte Carlo Risk Analysis")
        if st.button("Run Monte Carlo Simulation"):
            try:
                from pm_agent.risk.monte_carlo import run_monte_carlo_simulation
                from pm_agent.backtest.engine import BacktestConfig, CostModel
                from pm_agent.config import settings
                import plotly.express as px
                
                # Get signals
                signals_df = pd.read_sql("""
                    SELECT DISTINCT s.entity_id, s.ts, s.horizon_days
                    FROM signals s
                    ORDER BY s.ts
                    LIMIT 100
                """, engine)
                
                if len(signals_df) > 0:
                    config = BacktestConfig(
                        max_positions=settings.max_positions,
                        holding_days=settings.holding_period_days,
                        cost_model=CostModel(settings.cost_spread_bps, settings.cost_slippage_bps),
                    )
                    
                    with st.spinner("Running Monte Carlo simulation (this may take a minute)..."):
                        results = run_monte_carlo_simulation(signals_df, config, n_simulations=1000)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean Return", f"{results['mean_return']:.2%}")
                    col2.metric("95% VaR", f"{results['var_95']:.2%}")
                    col3.metric("Prob Profit", f"{results['prob_profit']:.1%}")
                    col4.metric("Expected Shortfall", f"{results['expected_shortfall']:.2%}")
                    
                    # Distribution plot
                    fig = px.histogram(
                        x=results['results'],
                        nbins=50,
                        title="Monte Carlo Return Distribution",
                        labels={'x': 'Final Equity', 'y': 'Frequency'}
                    )
                    fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Break Even")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No signals available for Monte Carlo simulation")
            except Exception as e:
                st.error(f"Monte Carlo simulation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Feature Importance
        import os
        import plotly.express as px
        
        feature_importance_path = "artifacts/feature_importance.csv"
        if os.path.exists(feature_importance_path):
            st.subheader("Feature Importance")
            fi_df = pd.read_csv(feature_importance_path)
            fig = px.bar(
                fi_df.sort_values('importance', ascending=True),
                x='importance',
                y='feature',
                orientation='h',
                title="ML Model Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available. Run training pipeline to generate.")
        
        # Feature Distributions
        st.subheader("Feature Statistics")
        try:
            features_df = pd.read_sql("SELECT * FROM features LIMIT 1000", engine)
            if len(features_df) > 0:
                feature_cols = ["delta_p_1d", "liquidity_score", "venue_disagreement"]
                available_cols = [c for c in feature_cols if c in features_df.columns]
                
                for col in available_cols:
                    with st.expander(f"Distribution: {col}"):
                        fig = px.histogram(features_df, x=col, title=f"Distribution: {col}")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature data available")
        except Exception as e:
            st.warning(f"Could not load features: {e}")
        
        # Calibration Plot (if backtest data exists)
        st.subheader("Model Calibration")
        try:
            cal_df = pd.read_sql("""
                SELECT 
                    s.entity_id,
                    s.ts,
                    s.strength as predicted_prob,
                    CASE WHEN bt.pnl > 0 THEN 1 ELSE 0 END as actual_outcome
                FROM signals s
                LEFT JOIN backtest_trades bt ON s.entity_id = bt.entity_id 
                    AND DATE(s.ts) = DATE(bt.entry_ts)
                WHERE s.strategy = 'ModelV1'
            """, engine)
            
            if len(cal_df) > 10:
                cal_df['prob_bin'] = pd.cut(cal_df['predicted_prob'], bins=10)
                calibration = cal_df.groupby('prob_bin').agg({
                    'predicted_prob': 'mean',
                    'actual_outcome': 'mean'
                }).reset_index()
                
                fig = px.scatter(
                    calibration,
                    x='predicted_prob',
                    y='actual_outcome',
                    title="Calibration Plot (Reliability Curve)"
                )
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(dash="dash", color="gray")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                brier = ((cal_df['predicted_prob'] - cal_df['actual_outcome']) ** 2).mean()
                st.metric("Brier Score", f"{brier:.4f}")
            else:
                st.info("Not enough data for calibration plot (need >10 signals with backtest results)")
        except Exception as e:
            st.warning(f"Could not generate calibration plot: {e}")
    
    with tab5:
        st.subheader("Advanced Features")
        
        # Market Regime Detection
        st.subheader("Market Regime Detection")
        if st.button("Detect Current Regime"):
            try:
                from pm_agent.features.regime import detect_market_regime, get_regime_adaptive_threshold
                from pm_agent.prices.provider import LocalCSVPriceProvider
                
                price_provider = LocalCSVPriceProvider()
                spy_prices = price_provider.load_prices("SPY")
                
                if len(spy_prices) > 50:
                    spy_returns = spy_prices["close"].pct_change().dropna()
                    regime = detect_market_regime(spy_returns)
                    
                    st.metric("Current Market Regime", regime.upper())
                    
                    # Show adaptive threshold
                    base_threshold = 0.08
                    adaptive_threshold = get_regime_adaptive_threshold(regime, base_threshold)
                    st.info(f"Recommended signal threshold for {regime} regime: {adaptive_threshold:.3f} (base: {base_threshold:.3f})")
                    
                    # Regime explanation
                    regime_info = {
                        "bull": "High returns, low volatility. Lower thresholds recommended.",
                        "bear": "Negative returns, high volatility. Higher thresholds recommended.",
                        "choppy": "Low returns, medium volatility. Very high thresholds recommended.",
                    }
                    st.write(regime_info.get(regime, "Unknown regime"))
                else:
                    st.warning("Not enough SPY data for regime detection (need >50 days)")
            except Exception as e:
                st.error(f"Regime detection failed: {e}")
        
        # Arbitrage Detection
        st.subheader("Cross-Venue Arbitrage Detection")
        if st.button("Scan for Arbitrage Opportunities"):
            try:
                from pm_agent.arbitrage.detector import ArbitrageDetector
                
                markets = pd.read_sql("""
                    SELECT m.market_id, m.event_id, m.venue_id, m.probability, m.title
                    FROM markets m
                    WHERE m.probability IS NOT NULL
                    ORDER BY m.updated_at DESC
                """, engine)
                
                if len(markets) > 0:
                    detector = ArbitrageDetector(min_spread=0.03)
                    
                    markets_list = [
                        {
                            "event_id": r["event_id"],
                            "venue_id": r["venue_id"],
                            "probability": float(r["probability"]) if r["probability"] else None,
                        }
                        for _, r in markets.iterrows()
                    ]
                    
                    opportunities = detector.find_arbitrage_opportunities(markets_list)
                    
                    if opportunities:
                        st.success(f"Found {len(opportunities)} arbitrage opportunities!")
                        
                        opp_df = pd.DataFrame([
                            {
                                "Event ID": opp.event_id,
                                "Venue 1": opp.venue_1,
                                "Venue 2": opp.venue_2,
                                "Prob 1": f"{opp.prob_1:.2%}",
                                "Prob 2": f"{opp.prob_2:.2%}",
                                "Spread": f"{opp.spread:.2%}",
                                "Expected Profit": f"{opp.expected_profit:.2%}",
                                "Action": opp.action,
                            }
                            for opp in opportunities
                        ])
                        st.dataframe(opp_df, use_container_width=True)
                    else:
                        st.info("No arbitrage opportunities found (spread < 3%)")
                else:
                    st.warning("No markets with probabilities available")
            except Exception as e:
                st.error(f"Arbitrage detection failed: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Sentiment Analysis
        st.subheader("News Sentiment Analysis")
        ticker_input = st.text_input("Enter ticker symbol", value="AAPL")
        if st.button("Analyze Sentiment"):
            try:
                from pm_agent.features.sentiment import SentimentAnalyzer
                import asyncio
                
                analyzer = SentimentAnalyzer(use_finbert=False)  # Set to True if transformers installed
                
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                sentiment_score = loop.run_until_complete(analyzer.get_news_sentiment(ticker_input, lookback_hours=24))
                loop.close()
                
                st.metric("Sentiment Score", f"{sentiment_score:.3f}")
                st.info("Score range: -1.0 (very negative) to +1.0 (very positive)")
                
                if sentiment_score > 0.3:
                    st.success("Positive sentiment detected")
                elif sentiment_score < -0.3:
                    st.error("Negative sentiment detected")
                else:
                    st.info("Neutral sentiment")
            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")
                st.info("Note: For advanced sentiment analysis, install: pip install transformers torch")
except Exception as e:
    st.error(f"Error connecting to database: {e}")

