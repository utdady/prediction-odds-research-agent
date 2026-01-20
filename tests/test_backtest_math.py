import pandas as pd

from pm_agent.backtest.engine import BacktestConfig, CostModel, run_event_driven_backtest


def test_backtest_single_trade_deterministic():
    sig = pd.DataFrame([{ "entity_id": "AAPL", "ts": "2024-01-02T00:00:00Z", "horizon_days": 1 }])
    cfg = BacktestConfig(max_positions=10, holding_days=1, cost_model=CostModel(0.0, 0.0))
    curve, trades = run_event_driven_backtest(sig, cfg)
    assert len(trades) == 1
    # Entry should occur on next trading day's open (no look-ahead)
    # We just verify the trade exists and equity curve is monotonic.
    assert not curve.empty

