from __future__ import annotations

import asyncio
import json
import uuid
from datetime import timedelta

import pandas as pd

from pm_agent.config import settings
from pm_agent.db import get_session
from pm_agent.logging import configure_logging
from pm_agent.backtest.engine import (
    BacktestConfig,
    CostModel,
    max_drawdown,
    run_event_driven_backtest,
    sharpe,
    sortino,
)
from pm_agent.sql import execute, fetch_all

import structlog


log = structlog.get_logger(__name__)


async def train_model_on_window(session, train_start, train_end) -> None:
    """Placeholder hook: for now we just log the window.

    In a more advanced setup, this would retrain and persist a model per window.
    Our current pipelines/train_model.py trains on all history, so we just
    document the intended behavior here.
    """
    log.info(
        "walk_forward_train_window",
        train_start=str(train_start),
        train_end=str(train_end),
    )


async def generate_signals_on_window(session, test_start, test_end) -> pd.DataFrame:
    """Return signals restricted to a test window."""
    rows = await fetch_all(
        session,
        """
        SELECT entity_id, ts, horizon_days
        FROM signals
        WHERE ts >= :start AND ts <= :end
        ORDER BY ts
        """,
        {"start": test_start, "end": test_end},
    )
    if not rows:
        return pd.DataFrame(columns=["entity_id", "ts", "horizon_days"])

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def aggregate_walk_forward_results(all_curves: list[pd.DataFrame]) -> pd.DataFrame:
    if not all_curves:
        return pd.DataFrame(columns=["date", "equity"])
    curve = (
        pd.concat(all_curves)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return curve


async def run_walk_forward() -> str | None:
    """Run walk-forward backtest with rolling train/test splits."""
    configure_logging()

    async with get_session() as session:
        feature_ts_rows = await fetch_all(
            session,
            "SELECT DISTINCT ts FROM features ORDER BY ts",
        )
        if not feature_ts_rows:
            log.info("walk_forward_no_features")
            return None

        dates = pd.to_datetime([r["ts"] for r in feature_ts_rows], utc=True)
        train_window = timedelta(days=settings.walk_train_days)
        test_window = timedelta(days=settings.walk_test_days)
        purge_gap = timedelta(days=settings.walk_purge_days)

        start_date = dates.min() + train_window
        end_date = dates.max()

        cfg = BacktestConfig(
            max_positions=settings.max_positions,
            holding_days=settings.holding_period_days,
            cost_model=CostModel(settings.cost_spread_bps, settings.cost_slippage_bps),
        )

        per_window_curves: list[pd.DataFrame] = []
        all_trades = []
        current_date = start_date

        while current_date < end_date:
            train_start = current_date - train_window
            train_end = current_date - purge_gap
            test_start = current_date
            test_end = min(current_date + test_window, end_date)

            log.info(
                "walk_forward_window",
                train=f"{train_start} to {train_end}",
                test=f"{test_start} to {test_end}",
            )

            await train_model_on_window(session, train_start, train_end)
            df_signals = await generate_signals_on_window(session, test_start, test_end)
            if df_signals.empty:
                current_date += timedelta(days=30)
                continue

            curve, trades = run_event_driven_backtest(df_signals, cfg)
            if not curve.empty:
                per_window_curves.append(curve)
            all_trades.extend(trades)

            current_date += timedelta(days=30)

        if not all_trades or not per_window_curves:
            log.info("walk_forward_no_trades")
            return None

        full_curve = aggregate_walk_forward_results(per_window_curves)
        full_curve = full_curve.set_index("date")
        daily = full_curve["equity"].pct_change().dropna()

        run_id = str(uuid.uuid4())

        await execute(
            session,
            "INSERT INTO backtest_runs(run_id, config, model_version, notes) VALUES (:run_id, :config::jsonb, :model_version, :notes)",
            {
                "run_id": run_id,
                "config": json.dumps(
                    {
                        "mode": "walk_forward",
                        "max_positions": cfg.max_positions,
                        "holding_days": cfg.holding_days,
                    }
                ),
                "model_version": "rule_v1",
                "notes": "walk_forward",
            },
        )

        for t in all_trades:
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
            "cagr": float(full_curve["equity"].iloc[-1] ** (252 / max(1, len(full_curve))) - 1.0),
            "sharpe": sharpe(daily),
            "sortino": sortino(daily),
            "max_drawdown": max_drawdown(full_curve["equity"]),
            "turnover": float(len(all_trades)),
            "hit_rate": float(sum(1 for t in all_trades if t.pnl > 0) / max(1, len(all_trades))),
            "avg_win": float(
                pd.Series([t.pnl for t in all_trades if t.pnl > 0]).mean()
                if any(t.pnl > 0 for t in all_trades)
                else 0.0
            ),
            "avg_loss": float(
                pd.Series([t.pnl for t in all_trades if t.pnl <= 0]).mean()
                if any(t.pnl <= 0 for t in all_trades)
                else 0.0
            ),
            "brier": None,
            "meta": {"n_trades": len(all_trades), "mode": "walk_forward"},
        }

        await execute(
            session,
            """
            INSERT INTO backtest_metrics(run_id, cagr, sharpe, sortino, max_drawdown, turnover, hit_rate, avg_win, avg_loss, brier, meta)
            VALUES (:run_id, :cagr, :sharpe, :sortino, :max_drawdown, :turnover, :hit_rate, :avg_win, :avg_loss, :brier, :meta::jsonb)
            ON CONFLICT (run_id) DO UPDATE SET meta=EXCLUDED.meta
            """,
            {**m, "run_id": run_id, "meta": json.dumps(m["meta"])},
        )

        await session.commit()

    from pathlib import Path

    Path("artifacts").mkdir(exist_ok=True)
    full_curve.to_csv(f"artifacts/equity_curve_walk_{run_id}.csv")
    log.info("run_backtest_walk_forward_done", run_id=run_id, n_trades=len(all_trades))
    return run_id


if __name__ == "__main__":
    asyncio.run(run_walk_forward())


