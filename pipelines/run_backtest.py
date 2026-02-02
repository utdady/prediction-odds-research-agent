from __future__ import annotations

import asyncio
import json
import uuid

import pandas as pd

from pm_agent.config import settings
from pm_agent.db import get_session
from pm_agent.logging import configure_logging
from pm_agent.backtest.engine import BacktestConfig, CostModel, max_drawdown, run_event_driven_backtest, sharpe, sortino
from pm_agent.sql import execute, fetch_all

import structlog


log = structlog.get_logger(__name__)


async def run() -> str | None:
    configure_logging()

    async with get_session() as session:
        sig = await fetch_all(session, "SELECT entity_id, ts, horizon_days FROM signals ORDER BY ts")
        if not sig:
            log.info("run_backtest_no_signals")
            return None

        df = pd.DataFrame(sig)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        cfg = BacktestConfig(
            max_positions=settings.max_positions,
            holding_days=settings.holding_period_days,
            cost_model=CostModel(settings.cost_spread_bps, settings.cost_slippage_bps),
        )

        curve, trades = run_event_driven_backtest(df, cfg)
        if curve.empty:
            log.info("run_backtest_empty_curve")
            return None

        curve = curve.set_index("date")
        daily = curve["equity"].pct_change().dropna()

        run_id = str(uuid.uuid4())

        await execute(
            session,
            "INSERT INTO backtest_runs(run_id, config, model_version, notes) VALUES (:run_id, CAST(:config AS jsonb), :model_version, :notes)",
            {"run_id": run_id, "config": json.dumps({"max_positions": cfg.max_positions, "holding_days": cfg.holding_days, "cost_model": {"spread_bps": cfg.cost_model.spread_bps, "slippage_bps": cfg.cost_model.slippage_bps}}), "model_version": "rule_v1", "notes": "mock"},
        )

        for t in trades:
            await execute(
                session,
                """
                INSERT INTO backtest_trades(run_id, entity_id, entry_ts, exit_ts, side, qty, entry_px, exit_px, cost_bps, pnl, pnl_pct)
                VALUES (:run_id, :entity_id, :entry_ts, :exit_ts, :side, :qty, :entry_px, :exit_px, :cost_bps, :pnl, :pnl_pct)
                """,
                {
                    "run_id": run_id,
                    "entity_id": t.entity_id,
                    "entry_ts": t.entry_ts,
                    "exit_ts": t.exit_ts,
                    "side": t.side,
                    "qty": t.qty,
                    "entry_px": t.entry_px,
                    "exit_px": t.exit_px,
                    "cost_bps": t.cost_bps,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                },
            )

        m = {
            "cagr": float(curve["equity"].iloc[-1] ** (252 / max(1, len(curve))) - 1.0),
            "sharpe": sharpe(daily),
            "sortino": sortino(daily),
            "max_drawdown": max_drawdown(curve["equity"]),
            "turnover": float(len(trades)),
            "hit_rate": float(sum(1 for t in trades if t.pnl > 0) / max(1, len(trades))),
            "avg_win": float(pd.Series([t.pnl for t in trades if t.pnl > 0]).mean() if any(t.pnl > 0 for t in trades) else 0.0),
            "avg_loss": float(pd.Series([t.pnl for t in trades if t.pnl <= 0]).mean() if any(t.pnl <= 0 for t in trades) else 0.0),
            "brier": None,
            "meta": {"n_trades": len(trades)},
        }

        await execute(
            session,
            """
            INSERT INTO backtest_metrics(run_id, cagr, sharpe, sortino, max_drawdown, turnover, hit_rate, avg_win, avg_loss, brier, meta)
            VALUES (:run_id, :cagr, :sharpe, :sortino, :max_drawdown, :turnover, :hit_rate, :avg_win, :avg_loss, :brier, CAST(:meta AS jsonb))
            ON CONFLICT (run_id) DO UPDATE SET meta=EXCLUDED.meta
            """,
            {**m, "run_id": run_id, "meta": json.dumps(m["meta"])},
        )

        await session.commit()

    from pathlib import Path
    Path("artifacts").mkdir(exist_ok=True)
    curve.to_csv(f"artifacts/equity_curve_{run_id}.csv")
    log.info("run_backtest_done", run_id=run_id, n_trades=len(trades))
    return run_id


if __name__ == "__main__":
    asyncio.run(run())

