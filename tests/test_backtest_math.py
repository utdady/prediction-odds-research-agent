import pandas as pd

from pm_agent.backtest.engine import BacktestConfig, CostModel, run_event_driven_backtest


def test_backtest_single_trade_deterministic():
    sig = pd.DataFrame([{ "entity_id": "AAPL", "ts": "2024-01-02T00:00:00Z", "horizon_days": 1 }])
    cfg = BacktestConfig(max_positions=10, holding_days=1, cost_model=CostModel(0.0, 0.0))
    curve, trades = run_event_driven_backtest(sig, cfg)
    assert len(trades) == 1
    # AAPL open 190 -> close next day 190 => ~0 return
    assert abs(trades[0].pnl_pct) < 1e-6

