from __future__ import annotations

import os
import pandas as pd
import streamlit as st

import sqlalchemy

st.set_page_config(page_title="PM Odds Agent", layout="wide")

st.title("Prediction Market Odds → Equity Signals")

st.sidebar.header("DB")
db_url = st.sidebar.text_input("DATABASE_URL_SYNC", value="postgresql+psycopg://pm:pm@localhost:5432/pm_research")

try:
    engine = sqlalchemy.create_engine(db_url)

    # Data freshness indicator
    try:
        freshness_df = pd.read_sql("""
            SELECT 
                MAX(ts) as last_feature_ts,
                MAX(tick_ts) as last_tick_ts,
                MAX(created_at) as last_backtest_ts
            FROM (
                SELECT MAX(ts) as ts, NULL::timestamp as tick_ts, NULL::timestamp as created_at FROM features
                UNION ALL
                SELECT NULL, MAX(tick_ts), NULL FROM odds_ticks
                UNION ALL
                SELECT NULL, NULL, MAX(created_at) FROM backtest_runs
            ) subq
        """, engine)
        
        if len(freshness_df) > 0 and freshness_df.iloc[0]["last_feature_ts"]:
            last_update = pd.to_datetime(freshness_df.iloc[0]["last_feature_ts"])
            hours_ago = (pd.Timestamp.now() - last_update).total_seconds() / 3600
            if hours_ago > 24:
                st.warning(f"⚠️ Data may be stale: Last feature update was {hours_ago:.1f} hours ago")
            elif hours_ago > 6:
                st.info(f"ℹ️ Last feature update: {hours_ago:.1f} hours ago")
    except Exception:
        pass  # Don't fail dashboard if freshness check fails

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        ["Markets", "Signals", "Backtest", "Diagnostics", "Advanced", "Agent", "Market Intelligence", "Health"]
    )

    with tab1:
        st.subheader("Latest markets")
        df = pd.read_sql("SELECT market_id, venue_id, title, status, updated_at FROM markets ORDER BY updated_at DESC LIMIT 1000", engine)
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("Latest signals")
        df = pd.read_sql("SELECT signal_id, entity_id, ts, strategy, side, strength, horizon_days FROM signals ORDER BY ts DESC LIMIT 1000", engine)
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
        runs = pd.read_sql("SELECT r.run_id, r.created_at, m.sharpe, m.max_drawdown, m.cagr FROM backtest_runs r LEFT JOIN backtest_metrics m ON r.run_id=m.run_id ORDER BY r.created_at DESC LIMIT 100", engine)
        st.dataframe(runs, use_container_width=True)

        if len(runs):
            run_id = st.selectbox("Select run", runs["run_id"].tolist())
            trades = pd.read_sql("SELECT * FROM backtest_trades WHERE run_id=%s ORDER BY entry_ts", engine, params=(run_id,))
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
            if st.button("Run Monte Carlo Simulation", key="mc_backtest"):
                try:
                    from pm_agent.risk.monte_carlo import run_monte_carlo_simulation
                    from pm_agent.backtest.engine import BacktestConfig, CostModel
                    from pm_agent.config import settings
                    
                    # Get signals for this run
                    signals_df = pd.read_sql("""
                        SELECT DISTINCT s.entity_id, s.ts, s.horizon_days
                        FROM signals s
                        JOIN backtest_trades bt ON s.entity_id = bt.entity_id
                        WHERE bt.run_id = %s
                        ORDER BY s.ts
                    """, engine, params=(run_id,))
                    
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
            if st.button("Generate PDF Tear Sheet", key="tear_sheet"):
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
        
        if st.button("Run Custom Backtest", key="custom_backtest"):
            try:
                from pm_agent.backtest.engine import run_event_driven_backtest, BacktestConfig, CostModel, sharpe, max_drawdown
                from pm_agent.config import settings
                
                # Fetch signals based on custom parameters
                # Use parameterized queries to prevent SQL injection
                signals_df = pd.read_sql("""
                    SELECT DISTINCT s.entity_id, s.ts, s.horizon_days
                    FROM signals s
                    JOIN features f ON s.entity_id = f.entity_id AND DATE(s.ts) = DATE(f.ts)
                    WHERE f.delta_p_1d > %s
                      AND f.liquidity_score >= %s
                    ORDER BY s.ts
                """, engine, params=(delta_threshold, min_liquidity))
                
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
        if st.button("Run Monte Carlo Simulation", key="mc_diagnostics"):
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
                        x=results['simulation_results'],
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
        
        # Enhanced Calibration Plot (if backtest data exists)
        st.subheader("Model Calibration")
        try:
            cal_df = pd.read_sql("""
                SELECT 
                    s.entity_id,
                    s.ts,
                    s.strength as predicted_prob,
                    CASE WHEN bt.pnl > 0 THEN 1 ELSE 0 END as actual_outcome
                FROM signals s
                INNER JOIN backtest_trades bt ON s.entity_id = bt.entity_id 
                    AND DATE(s.ts) = DATE(bt.entry_ts)
                WHERE s.strategy = 'ModelV1'
            """, engine)
            
            # Filter for signals with actual backtest results
            cal_df = cal_df[cal_df['actual_outcome'].notna()]
            
            if len(cal_df) > 10:
                # Brier score decomposition
                from pm_agent.backtest.metrics import brier_score_decomposition
                import numpy as np
                
                y_true = cal_df['actual_outcome'].fillna(0).values
                y_pred = cal_df['predicted_prob'].fillna(0.5).values
                
                brier_decomp = brier_score_decomposition(y_true, y_pred)
                
                # Reliability diagram with confidence intervals
                cal_df['prob_bin'] = pd.cut(cal_df['predicted_prob'], bins=10)
                calibration = cal_df.groupby('prob_bin').agg({
                    'predicted_prob': ['mean', 'count'],
                    'actual_outcome': ['mean', 'std']
                }).reset_index()
                
                calibration.columns = ['prob_bin', 'pred_mean', 'count', 'obs_mean', 'obs_std']
                
                # Convert prob_bin to string to avoid categorical issues
                calibration['prob_bin'] = calibration['prob_bin'].astype(str)
                
                # Calculate confidence intervals (95%)
                calibration['ci_lower'] = calibration['obs_mean'] - 1.96 * calibration['obs_std'] / np.sqrt(calibration['count'])
                calibration['ci_upper'] = calibration['obs_mean'] + 1.96 * calibration['obs_std'] / np.sqrt(calibration['count'])
                
                # Fill NaN only on numeric columns (not on prob_bin which is now string)
                numeric_cols = ['pred_mean', 'count', 'obs_mean', 'obs_std', 'ci_lower', 'ci_upper']
                for col in numeric_cols:
                    if col in calibration.columns:
                        calibration[col] = calibration[col].fillna(0)
                
                # Remove any rows with invalid data
                calibration = calibration[calibration['count'] > 0].copy()
                
                # Reset index to ensure alignment
                calibration = calibration.reset_index(drop=True)
                
                # Calculate error bars directly from the filtered dataframe
                error_upper = (calibration['ci_upper'] - calibration['obs_mean']).fillna(0).values
                error_lower = (calibration['obs_mean'] - calibration['ci_lower']).fillna(0).values
                
                # Use go.Scatter directly for better control
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=calibration['pred_mean'].values,
                    y=calibration['obs_mean'].values,
                    mode='markers',
                    marker=dict(
                        size=calibration['count'].values * 2,  # Scale size for visibility
                        color=calibration['obs_mean'].values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Observed Frequency")
                    ),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error_upper,
                        arrayminus=error_lower,
                        visible=True
                    ),
                    name='Calibration Points',
                    text=[f"Count: {c}" for c in calibration['count'].values],
                    hovertemplate='Predicted: %{x:.3f}<br>Observed: %{y:.3f}<br>Count: %{text}<extra></extra>'
                ))
                
                # Add perfect calibration line
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(dash="dash", color="gray", width=2),
                    name="Perfect Calibration"
                )
                
                fig.update_layout(
                    title="Reliability Diagram with Confidence Intervals",
                    xaxis_title="Predicted Probability",
                    yaxis_title="Observed Frequency",
                    hovermode='closest'
                )
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(dash="dash", color="gray"),
                    name="Perfect Calibration"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Brier score metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Brier Score", f"{brier_decomp['brier']:.4f}")
                col2.metric("Uncertainty", f"{brier_decomp['uncertainty']:.4f}")
                col3.metric("Resolution", f"{brier_decomp['resolution']:.4f}")
                col4.metric("Calibration", f"{brier_decomp['calibration']:.4f}")
                
                st.caption("Brier = Uncertainty - Resolution + Calibration")
            else:
                st.info("Not enough data for calibration plot (need >10 signals with backtest results)")
        except Exception as e:
            st.warning(f"Could not generate calibration plot: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    with tab5:
        st.subheader("Advanced Features")
        
        # Market Regime Detection
        st.subheader("Market Regime Detection")
        if st.button("Detect Current Regime", key="detect_regime"):
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
        if st.button("Scan for Arbitrage Opportunities", key="scan_arbitrage"):
            try:
                from pm_agent.arbitrage.detector import ArbitrageDetector
                
                markets = pd.read_sql("""
                    SELECT DISTINCT ON (m.market_id)
                        m.market_id, 
                        m.event_id, 
                        m.venue_id, 
                        t.p_norm as probability, 
                        m.title
                    FROM markets m
                    INNER JOIN odds_ticks t ON m.market_id = t.market_id
                    WHERE t.p_norm IS NOT NULL
                    ORDER BY m.market_id, t.tick_ts DESC
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
        ticker_input = st.text_input("Enter ticker symbol", value="TSLA")
        
        col1, col2 = st.columns(2)
        with col1:
            lookback_hours = st.slider("Lookback hours", 1, 168, 24)
        with col2:
            use_finbert = st.checkbox("Use FinBERT (requires transformers)", value=False)
        
        if st.button("Analyze Sentiment", key="analyze_sentiment"):
            try:
                from pm_agent.features.sentiment import SentimentAnalyzer
                import asyncio
                
                with st.spinner(f"Analyzing sentiment for {ticker_input}..."):
                    analyzer = SentimentAnalyzer(use_finbert=use_finbert)
                    
                    # Run async function - use thread-based approach to avoid event loop conflicts
                    import threading
                    result_container = {"score": None, "error": None}
                    
                    def run_async():
                        try:
                            # Create a new event loop in this thread
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                result_container["score"] = loop.run_until_complete(
                                    analyzer.get_news_sentiment(ticker_input.upper(), lookback_hours=lookback_hours)
                                )
                            finally:
                                loop.close()
                        except Exception as e:
                            result_container["error"] = str(e)
                            import traceback
                            result_container["traceback"] = traceback.format_exc()
                    
                    thread = threading.Thread(target=run_async)
                    thread.start()
                    thread.join(timeout=30)  # 30 second timeout
                    
                    if thread.is_alive():
                        st.error("Sentiment analysis timed out after 30 seconds")
                        sentiment_score = 0.0
                    elif result_container["error"]:
                        st.error(f"Error fetching sentiment: {result_container['error']}")
                        if "traceback" in result_container:
                            with st.expander("Error details"):
                                st.code(result_container["traceback"])
                        sentiment_score = 0.0
                    elif result_container["score"] is not None:
                        sentiment_score = result_container["score"]
                    else:
                        st.warning("Sentiment analysis returned no result")
                        sentiment_score = 0.0
                
                st.metric("Sentiment Score", f"{sentiment_score:.3f}")
                st.caption("Score range: -1.0 (very negative) to +1.0 (very positive)")
                
                # Display sentiment interpretation with more granular feedback
                if sentiment_score > 0.5:
                    st.success(f"✅ Strongly Positive sentiment ({sentiment_score:.3f})")
                elif sentiment_score > 0.3:
                    st.success(f"✅ Positive sentiment ({sentiment_score:.3f})")
                elif sentiment_score > 0.1:
                    st.info(f"📈 Slightly Positive sentiment ({sentiment_score:.3f})")
                elif sentiment_score < -0.5:
                    st.error(f"❌ Strongly Negative sentiment ({sentiment_score:.3f})")
                elif sentiment_score < -0.3:
                    st.error(f"❌ Negative sentiment ({sentiment_score:.3f})")
                elif sentiment_score < -0.1:
                    st.info(f"📉 Slightly Negative sentiment ({sentiment_score:.3f})")
                else:
                    st.info(f"⚖️ Neutral sentiment ({sentiment_score:.3f})")
                    
                # Show note about FinBERT
                if not use_finbert:
                    st.caption("💡 Tip: Enable FinBERT for more accurate sentiment analysis (requires: pip install transformers torch)")
                    
                # Debug info (if score is 0, show why)
                if abs(sentiment_score) < 0.01:
                    with st.expander("ℹ️ Why is the score 0.0?"):
                        st.write("A score of 0.0 typically means:")
                        st.write("1. **No recent news articles found** - Try increasing the lookback hours")
                        st.write("2. **Articles had mixed/neutral sentiment** - No clear positive or negative keywords")
                        st.write("3. **News feed unavailable** - Google News RSS may be blocked or rate-limited")
                        st.write(f"4. **Network issues** - Check your internet connection")
                        st.write(f"\n**Current settings:**")
                        st.write(f"- Ticker: {ticker_input.upper()}")
                        st.write(f"- Lookback: {lookback_hours} hours")
                        st.write(f"- FinBERT: {'Enabled' if use_finbert else 'Disabled (using keyword-based)'}")
                    
                # Debug info (if score is 0, show why)
                if abs(sentiment_score) < 0.01:
                    st.warning("⚠️ Sentiment score is near zero. This could mean:")
                    st.write("- No recent news articles found")
                    st.write("- Articles had mixed/neutral sentiment")
                    st.write("- News feed may be unavailable")
                    st.write(f"Try increasing lookback hours (currently {lookback_hours}h) or check if news is available for {ticker_input}")
                    
            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                st.info("Note: For advanced sentiment analysis, install: pip install transformers torch")
    
    with tab6:
        st.subheader("🤖 Search Agent")
        st.write("Ask me anything about companies, signals, backtests, markets, or features!")

        # Initialize chat search agent (company/market/backtest queries)
        from pm_agent.agent.search_agent import SearchAgent
        agent = SearchAgent(engine)
        
        # Chat interface
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "data" in msg and msg["data"] is not None:
                    if isinstance(msg["data"], pd.DataFrame) and len(msg["data"]) > 0:
                        st.dataframe(msg["data"], use_container_width=True)
                    elif isinstance(msg["data"], dict):
                        # Company detail view
                        if "company" in msg["data"]:
                            st.write("**Company Info:**")
                            st.dataframe(msg["data"]["company"], use_container_width=True)
                        if "markets" in msg["data"] and len(msg["data"]["markets"]) > 0:
                            st.write("**Related Markets:**")
                            st.dataframe(msg["data"]["markets"], use_container_width=True)
                        if "signals" in msg["data"] and len(msg["data"]["signals"]) > 0:
                            st.write("**Signals:**")
                            st.dataframe(msg["data"]["signals"], use_container_width=True)
                        if "features" in msg["data"] and len(msg["data"]["features"]) > 0:
                            st.write("**Features:**")
                            st.dataframe(msg["data"]["features"], use_container_width=True)
        
        # User input
        user_query = st.chat_input("Ask about companies, signals, backtests, markets, or features...")
        
        if user_query:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Get agent response
            with st.chat_message("assistant"):
                try:
                    response = agent.search(user_query)
                    
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Display data if available
                    if response["data"] is not None:
                        if isinstance(response["data"], pd.DataFrame):
                            if len(response["data"]) > 0:
                                st.dataframe(response["data"], use_container_width=True)
                        elif isinstance(response["data"], dict):
                            # Company detail view
                            if "company" in response["data"]:
                                st.write("**Company Info:**")
                                st.dataframe(response["data"]["company"], use_container_width=True)
                            
                            if "markets" in response["data"] and len(response["data"]["markets"]) > 0:
                                st.write("**Related Markets:**")
                                st.dataframe(response["data"]["markets"], use_container_width=True)
                            
                            if "signals" in response["data"] and len(response["data"]["signals"]) > 0:
                                st.write("**Signals:**")
                                st.dataframe(response["data"]["signals"], use_container_width=True)
                            
                            if "features" in response["data"] and len(response["data"]["features"]) > 0:
                                st.write("**Features:**")
                                st.dataframe(response["data"]["features"], use_container_width=True)
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "data": response["data"],
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing query: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
                    import traceback
                    st.code(traceback.format_exc())
            
            st.rerun()
        
        # Quick action buttons
        st.subheader("Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Show All Companies", key="show_companies"):
                st.session_state.chat_history.append({"role": "user", "content": "show all companies"})
                st.rerun()
        
        with col2:
            if st.button("📈 Recent Signals", key="recent_signals"):
                st.session_state.chat_history.append({"role": "user", "content": "show signals"})
                st.rerun()
        
        with col3:
            if st.button("📉 Backtest Results", key="backtest_results"):
                st.session_state.chat_history.append({"role": "user", "content": "show backtest results"})
                st.rerun()
        
        with col4:
            if st.button("❓ Help", key="help_btn"):
                st.session_state.chat_history.append({"role": "user", "content": "help"})
                st.rerun()

        # Clear chat button
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.write("""
            - **"show AAPL"** - Get all information about Apple
            - **"signals for TSLA"** - Get trading signals for Tesla
            - **"backtest performance"** - See strategy performance
            - **"markets for MSFT"** - List markets for Microsoft
            - **"features for AAPL"** - Show feature data for Apple
            - **"what companies do we have?"** - List all companies
            - **"help"** - Show help message
            """)

    # ------------------------------------------------------------------
    # Market Intelligence tab - uses StockAnalysisAgent
    # ------------------------------------------------------------------
    with tab7:
        st.subheader("📊 Market Intelligence")
        st.write("AI-powered stock analysis based on prediction market odds")

        from pm_agent.agent.stock_analysis import StockAnalysisAgent

        analyst = StockAnalysisAgent(engine)

        analysis_type = st.selectbox(
            "Choose Analysis Type",
            ["Daily Digest", "Stock Performance", "Individual Stock", "Compare Stocks", "Sector Analysis"],
        )

        if analysis_type == "Daily Digest":
            st.subheader("📰 Daily Market Digest")
            if st.button("Generate Today's Digest", key="daily_digest"):
                with st.spinner("Analyzing market data..."):
                    digest = analyst.get_daily_digest()
                    st.markdown(digest)

        elif analysis_type == "Stock Performance":
            st.subheader("📈 Stock Performance Analysis")
            col1, col2 = st.columns(2)
            with col1:
                lookback = st.slider("Lookback Period (days)", 1, 30, 7)
            with col2:
                threshold = st.slider("Significance Threshold", 0.01, 0.20, 0.08, 0.01)

            if st.button("Analyze All Stocks", key="analyze_all_stocks"):
                with st.spinner("Analyzing stocks..."):
                    results = analyst.analyze_stock_performance(
                        ticker=None,
                        lookback_days=lookback,
                        threshold=threshold,
                    )
                    st.markdown(results["analysis"])

                    if results["performing_well"]:
                        st.subheader("🚀 Top Performers")
                        well_df = pd.DataFrame(results["performing_well"])
                        st.dataframe(
                            well_df[
                                [
                                    "ticker",
                                    "name",
                                    "current_probability",
                                    "total_change",
                                    "avg_daily_change",
                                    "volatility",
                                    "liquidity_note",
                                ]
                            ],
                            use_container_width=True,
                        )

                    if results["performing_poorly"]:
                        st.subheader("📉 Under Pressure")
                        poor_df = pd.DataFrame(results["performing_poorly"])
                        st.dataframe(
                            poor_df[
                                [
                                    "ticker",
                                    "name",
                                    "current_probability",
                                    "total_change",
                                    "avg_daily_change",
                                    "volatility",
                                    "liquidity_note",
                                ]
                            ],
                            use_container_width=True,
                        )

                    if results["neutral"]:
                        with st.expander(f"⚖️ Neutral Stocks ({len(results['neutral'])})"):
                            neutral_df = pd.DataFrame(results["neutral"])
                            st.dataframe(
                                neutral_df[
                                    [
                                        "ticker",
                                        "name",
                                        "current_probability",
                                        "total_change",
                                        "liquidity_note",
                                    ]
                                ],
                                use_container_width=True,
                            )

        elif analysis_type == "Individual Stock":
            st.subheader("🔍 Individual Stock Analysis")
            tickers_df = pd.read_sql("SELECT DISTINCT ticker FROM entities ORDER BY ticker", engine)
            available_tickers = tickers_df["ticker"].tolist()
            ticker = st.selectbox("Select Stock", available_tickers)

            if st.button("Get Stock Outlook", key="get_stock_outlook"):
                with st.spinner(f"Analyzing {ticker}..."):
                    outlook = analyst.get_stock_outlook(ticker)
                    if outlook["outlook"] != "unknown":
                        st.markdown(f"# {outlook['ticker']} - {outlook['name']}")
                        st.markdown(f"**Sector**: {outlook['sector']}")
                        st.markdown("---")

                        col1, col2, col3, col4 = st.columns(4)
                        outlook_emoji = {"bullish": "🚀", "bearish": "📉", "neutral": "⚖️"}
                        with col1:
                            st.metric("Outlook", outlook["outlook"].title(), help=outlook_emoji.get(outlook["outlook"], ""))
                        with col2:
                            st.metric("Confidence", f"{outlook['confidence']:.0%}")
                        with col3:
                            st.metric("Risk Level", outlook["risk_level"])
                        with col4:
                            st.metric(
                                "Probability",
                                f"{outlook['details']['current_probability']:.1%}",
                                delta=f"{outlook['details']['total_change']:+.1%}",
                            )

                        st.markdown("---")
                        rec = outlook["recommendation"]
                        if "Strong Buy" in rec:
                            st.success(f"**Recommendation**: {rec}")
                        elif "Buy" in rec:
                            st.info(f"**Recommendation**: {rec}")
                        elif "Sell" in rec:
                            st.warning(f"**Recommendation**: {rec}")
                        else:
                            st.info(f"**Recommendation**: {rec}")

                        st.subheader("📊 Details")
                        dcol1, dcol2 = st.columns(2)
                        with dcol1:
                            st.write(f"**Total Change**: {outlook['details']['total_change']:+.2%}")
                            st.write(f"**Avg Daily Change**: {outlook['details']['avg_daily_change']:+.2%}")
                            st.write(f"**Trend**: {outlook['details']['trend_direction'].title()}")
                        with dcol2:
                            st.write(f"**Volatility**: {outlook['details']['volatility']:.3f}")
                            st.write(f"**Liquidity**: {outlook['details']['liquidity_note']}")
                            st.write(f"**Reason**: {outlook['details']['reason']}")

                        if outlook["recent_signals"]:
                            st.subheader("📡 Recent Signals")
                            st.dataframe(pd.DataFrame(outlook["recent_signals"]), use_container_width=True)
                        else:
                            st.info("No recent trading signals for this stock")
                    else:
                        st.warning(f"No data available for {ticker}")

        elif analysis_type == "Compare Stocks":
            st.subheader("⚖️ Stock Comparison")
            tickers_df = pd.read_sql("SELECT DISTINCT ticker FROM entities ORDER BY ticker", engine)
            available_tickers = tickers_df["ticker"].tolist()
            selected_tickers = st.multiselect(
                "Select Stocks to Compare (2-5 stocks)",
                available_tickers,
                default=available_tickers[:3] if len(available_tickers) >= 3 else available_tickers,
            )

            if len(selected_tickers) < 2:
                st.warning("Please select at least 2 stocks to compare")
            elif len(selected_tickers) > 5:
                st.warning("Please select at most 5 stocks to compare")
            elif st.button("Compare Stocks", key="compare_stocks"):
                with st.spinner("Comparing stocks..."):
                    comparison = analyst.compare_stocks(selected_tickers)
                    if comparison["comparison"]:
                        st.markdown(comparison["summary"])
                        st.markdown("---")
                        comp_data = []
                        for stock in comparison["comparison"]:
                            comp_data.append(
                                {
                                    "Ticker": stock["ticker"],
                                    "Name": stock["name"],
                                    "Outlook": stock["outlook"].title(),
                                    "Recommendation": stock["recommendation"],
                                    "Confidence": f"{stock['confidence']:.0%}",
                                    "Probability": f"{stock['details']['current_probability']:.1%}",
                                    "Change": f"{stock['details']['total_change']:+.1%}",
                                    "Risk": stock["risk_level"],
                                }
                            )
                        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        elif analysis_type == "Sector Analysis":
            st.subheader("🏭 Sector Performance Analysis")
            if st.button("Analyze Sectors", key="analyze_sectors"):
                with st.spinner("Analyzing sectors..."):
                    sector_results = analyst.get_sector_analysis()
                    if sector_results["sectors"]:
                        st.markdown(sector_results["analysis"])
                        st.markdown("---")
                        sector_df = pd.DataFrame(sector_results["sectors"])
                        sector_df["avg_change"] = sector_df["avg_change"].apply(lambda x: f"{x:+.2%}")
                        sector_df["avg_probability"] = sector_df["avg_probability"].apply(lambda x: f"{x:.1%}")
                        sector_df["liquidity"] = sector_df["liquidity"].apply(lambda x: f"{x:.0%}")
                        st.dataframe(
                            sector_df[
                                ["emoji", "sector", "sentiment", "avg_change", "avg_probability", "n_stocks", "liquidity"]
                            ],
                            use_container_width=True,
                        )
    
    with tab8:
        st.subheader("🏥 System Health")
        st.write("Monitor system status, data freshness, and component health")
        
        # Health checks
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Database", "✅ Connected" if engine else "❌ Disconnected")
        
        # Data freshness
        try:
            freshness = pd.read_sql("""
                SELECT 
                    MAX(ts) as last_feature_ts,
                    MAX(tick_ts) as last_tick_ts,
                    COUNT(*) as feature_count
                FROM (
                    SELECT MAX(ts) as ts, NULL::timestamp as tick_ts, COUNT(*) as cnt FROM features
                    UNION ALL
                    SELECT NULL, MAX(tick_ts), 0 FROM odds_ticks
                ) subq
            """, engine)
            
            if len(freshness) > 0:
                last_feature = freshness.iloc[0]["last_feature_ts"]
                last_tick = freshness.iloc[0]["last_tick_ts"]
                
                if last_feature:
                    last_feature_ts = pd.to_datetime(last_feature)
                    now_ts = pd.Timestamp.now(tz=last_feature_ts.tz) if last_feature_ts.tz else pd.Timestamp.now()
                    hours_ago = (now_ts - last_feature_ts).total_seconds() / 3600
                    with col2:
                        status = "✅ Fresh" if hours_ago < 6 else "⚠️ Stale" if hours_ago < 24 else "❌ Very Stale"
                        st.metric("Features", status, f"{hours_ago:.1f} hours ago")
                
                if last_tick:
                    last_tick_ts = pd.to_datetime(last_tick)
                    now_ts = pd.Timestamp.now(tz=last_tick_ts.tz) if last_tick_ts.tz else pd.Timestamp.now()
                    tick_hours_ago = (now_ts - last_tick_ts).total_seconds() / 3600
                    with col3:
                        tick_status = "✅ Fresh" if tick_hours_ago < 1 else "⚠️ Stale" if tick_hours_ago < 6 else "❌ Very Stale"
                        st.metric("Ticks", tick_status, f"{tick_hours_ago:.1f} hours ago")
        except Exception as e:
            st.error(f"Error checking freshness: {e}")
        
        # Orchestrator state
        st.subheader("📊 Pipeline Status")
        try:
            state_df = pd.read_sql("""
                SELECT 
                    component,
                    last_run_at,
                    is_dirty
                FROM orchestrator_state
                ORDER BY component
            """, engine)
            
            if len(state_df) > 0:
                # Format timestamps
                if "last_run_at" in state_df.columns:
                    state_df["last_run_at"] = pd.to_datetime(state_df["last_run_at"]).dt.strftime("%Y-%m-%d %H:%M")
                
                # Color code dirty status
                def format_status(row):
                    if row.get("is_dirty"):
                        return "🔴 Dirty"
                    elif row.get("last_run_at") and pd.notna(row.get("last_run_at")):
                        return "✅ Clean"
                    else:
                        return "⚪ Never Run"
                
                state_df["Status"] = state_df.apply(format_status, axis=1)
                st.dataframe(state_df, use_container_width=True)
            else:
                st.info("No orchestrator state found. Run pipelines to populate.")
        except Exception as e:
            st.warning(f"Could not load orchestrator state: {e}")
        
        # Data quality logs
        st.subheader("🔍 Data Quality Issues")
        try:
            # Default to a shorter lookback so old mock-data issues don't dominate the UI.
            lookback_hours = st.selectbox(
                "Lookback window",
                options=[1, 6, 24, 72, 168],
                index=2,  # 24h
                key="dq_lookback_hours",
                help="How far back to show data quality issues.",
            )

            # Use string formatting for interval (safe since lookback_hours is validated integer from selectbox)
            dq_df = pd.read_sql(
                f"""
                SELECT 
                    scope,
                    level,
                    message,
                    ts as created_at,
                    context
                FROM data_quality_log
                WHERE ts >= NOW() - INTERVAL '{int(lookback_hours)} hours'
                ORDER BY ts DESC
                LIMIT 200
                """,
                engine,
            )
            
            if len(dq_df) > 0:
                st.dataframe(dq_df, use_container_width=True)
                
                # Summary
                error_count = int((dq_df["level"] == "error").sum())
                warning_count = int((dq_df["level"] == "warning").sum())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Errors ({lookback_hours}h)", error_count)
                with col2:
                    st.metric(f"Warnings ({lookback_hours}h)", warning_count)
            else:
                st.success(f"✅ No data quality issues in the last {lookback_hours} hours")
        except Exception as e:
            st.warning(f"Could not load data quality logs: {e}")
        
        # Model status
        st.subheader("🤖 Model Status")
        from pathlib import Path
        
        model_path = Path("artifacts/model_v1.joblib")
        if model_path.exists():
            st.success(f"✅ Model available: {model_path}")
            try:
                import joblib
                model_data = joblib.load(model_path)
                if "features" in model_data:
                    st.write(f"**Features**: {', '.join(model_data['features'])}")
            except Exception as e:
                st.warning(f"Could not load model metadata: {e}")
        else:
            st.warning("⚠️ Model not found. Run training pipeline to generate model.")

except Exception as e:
    st.error(f"Error connecting to database: {e}")

