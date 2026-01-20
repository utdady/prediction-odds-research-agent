import pandas as pd

from pm_agent.backtest.engine import BacktestConfig, CostModel, run_event_driven_backtest


def test_backtest_single_trade_deterministic():
    sig = pd.DataFrame([{ "entity_id": "AAPL", "ts": "2024-01-02T00:00:00Z", "horizon_days": 1 }])
    cfg = BacktestConfig(max_positions=10, holding_days=1, cost_model=CostModel(0.0, 0.0))
    curve, trades = run_event_driven_backtest(sig, cfg)
    assert len(trades) == 1
    # Entry should occur after the signal date (no look-ahead)
    signal_date = pd.Timestamp("2024-01-02").date()
    entry_date = trades[0].entry_ts.date()
    assert entry_date > signal_date

