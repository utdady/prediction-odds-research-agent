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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Markets", "Signals", "Backtest", "Diagnostics", "Advanced", "Agent", "Market Intelligence"]
    )

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
                LEFT JOIN backtest_trades bt ON s.entity_id = bt.entity_id 
                    AND DATE(s.ts) = DATE(bt.entry_ts)
                WHERE s.strategy = 'ModelV1'
            """, engine)
            
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
                
                # Calculate confidence intervals (95%)
                calibration['ci_lower'] = calibration['obs_mean'] - 1.96 * calibration['obs_std'] / np.sqrt(calibration['count'])
                calibration['ci_upper'] = calibration['obs_mean'] + 1.96 * calibration['obs_std'] / np.sqrt(calibration['count'])
                calibration = calibration.fillna(0)
                
                # Reliability diagram
                fig = px.scatter(
                    calibration,
                    x='pred_mean',
                    y='obs_mean',
                    size='count',
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=calibration['ci_upper'] - calibration['obs_mean'],
                        arrayminus=calibration['obs_mean'] - calibration['ci_lower']
                    ),
                    title="Reliability Diagram with Confidence Intervals",
                    labels={'pred_mean': 'Predicted Probability', 'obs_mean': 'Observed Frequency'}
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
            if st.button("📊 Show All Companies"):
                st.session_state.chat_history.append({"role": "user", "content": "show all companies"})
                st.rerun()
        
        with col2:
            if st.button("📈 Recent Signals"):
                st.session_state.chat_history.append({"role": "user", "content": "show signals"})
                st.rerun()
        
        with col3:
            if st.button("📉 Backtest Results"):
                st.session_state.chat_history.append({"role": "user", "content": "show backtest results"})
                st.rerun()
        
        with col4:
            if st.button("❓ Help"):
                st.session_state.chat_history.append({"role": "user", "content": "help"})
                st.rerun()

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
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
            if st.button("Generate Today's Digest"):
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

            if st.button("Analyze All Stocks"):
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

            if st.button("Get Stock Outlook"):
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
            elif st.button("Compare Stocks"):
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
            if st.button("Analyze Sectors"):
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

except Exception as e:
    st.error(f"Error connecting to database: {e}")

