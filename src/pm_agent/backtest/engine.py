from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from pm_agent.prices.provider import LocalCSVPriceProvider


@dataclass(frozen=True)
class CostModel:
    spread_bps: float
    slippage_bps: float

    def total_bps(self) -> float:
        return float(self.spread_bps + self.slippage_bps)


@dataclass(frozen=True)
class BacktestConfig:
    max_positions: int = 10
    holding_days: int = 5
    cost_model: CostModel = CostModel(5.0, 5.0)


@dataclass(frozen=True)
class Trade:
    entity_id: str
    entry_ts: datetime
    exit_ts: datetime
    side: str
    qty: float
    entry_px: float
    exit_px: float
    cost_bps: float
    pnl: float
    pnl_pct: float


def _apply_cost(px: float, cost_bps: float, side: str, is_entry: bool) -> float:
    # Simple: entry worse, exit worse
    mult = 1.0 + (cost_bps / 1e4)
    if side == "LONG":
        return px * mult if is_entry else px / mult
    raise ValueError("v1 supports LONG only")


def run_event_driven_backtest(signals: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, list[Trade]]:
    price = LocalCSVPriceProvider()
    cost_bps = config.cost_model.total_bps()

    # signals: columns [entity_id, ts, horizon_days]
    signals = signals.sort_values("ts")

    equity = 1.0
    curve = []
    trades: list[Trade] = []

    for _, s in signals.iterrows():
        ticker = s["entity_id"]
        ts = pd.Timestamp(s["ts"], tz="UTC").tz_convert(None).normalize()
        px = price.load_prices(ticker)

        if ts not in px.index:
            continue
        idx = list(px.index)
        i = idx.index(ts)
        j = min(i + int(s.get("horizon_days", config.holding_days)), len(idx) - 1)
        entry_px = float(px.iloc[i]["open"])
        exit_px = float(px.iloc[j]["close"])

        entry_px_eff = _apply_cost(entry_px, cost_bps, "LONG", True)
        exit_px_eff = _apply_cost(exit_px, cost_bps, "LONG", False)

        ret = (exit_px_eff / entry_px_eff) - 1.0
        weight = 1.0 / config.max_positions
        pnl_pct = weight * ret
        equity *= (1.0 + pnl_pct)

        trades.append(
            Trade(
                entity_id=ticker,
                entry_ts=ts.to_pydatetime(),
                exit_ts=idx[j].to_pydatetime(),
                side="LONG",
                qty=weight,
                entry_px=entry_px_eff,
                exit_px=exit_px_eff,
                cost_bps=cost_bps,
                pnl=pnl_pct,
                pnl_pct=pnl_pct,
            )
        )
        curve.append({"date": idx[j], "equity": equity})

    curve_df = pd.DataFrame(curve).drop_duplicates(subset=["date"]).sort_values("date")
    return curve_df, trades


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0


def sharpe(daily_returns: pd.Series, ann: int = 252) -> float:
    if daily_returns.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(ann) * daily_returns.mean() / daily_returns.std(ddof=1))


def sortino(daily_returns: pd.Series, ann: int = 252) -> float:
    downside = daily_returns[daily_returns < 0]
    if downside.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(ann) * daily_returns.mean() / downside.std(ddof=1))

