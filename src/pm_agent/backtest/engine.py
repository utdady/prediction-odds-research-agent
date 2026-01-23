from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd

from pm_agent.backtest.exit_rules import ExitRuleManager, Position
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
    use_exit_rules: bool = True
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = -0.03


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
    raise ValueError(f"Backtest engine v1 only supports LONG positions. Received side='{side}'. SHORT positions not yet implemented.")


def run_event_driven_backtest(signals: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, list[Trade]]:
    price = LocalCSVPriceProvider()
    cost_bps = config.cost_model.total_bps()
    
    # Initialize exit rules if enabled
    exit_rules = None
    if config.use_exit_rules:
        exit_rules = ExitRuleManager(
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            trailing_stop_pct=config.trailing_stop_pct,
        )

    # signals: columns [entity_id, ts, horizon_days]
    signals = signals.sort_values("ts")

    equity = 1.0
    curve = []
    trades: list[Trade] = []
    open_positions: dict[str, Position] = {}  # Track open positions for exit rules

    def _next_trading_day(d: date) -> date:
        nxt = d + timedelta(days=1)
        while nxt.weekday() >= 5:  # skip Sat/Sun
            nxt += timedelta(days=1)
        return nxt

    for _, s in signals.iterrows():
        ticker = s["entity_id"]
        ts = pd.Timestamp(s["ts"], tz="UTC").tz_convert(None).normalize()
        signal_date = ts.date()
        entry_date = _next_trading_day(signal_date)
        entry_ts = pd.Timestamp(entry_date)
        px = price.load_prices(ticker)

        # use next-day open as entry
        if entry_ts not in px.index:
            continue
        idx = list(px.index)
        i = idx.index(entry_ts)
        max_hold_days = int(s.get("horizon_days", config.holding_days))
        j_target = min(i + max_hold_days, len(idx) - 1)
        
        entry_px = float(px.iloc[i]["open"])
        entry_px_eff = _apply_cost(entry_px, cost_bps, "LONG", True)
        
        # Check for early exit if exit rules enabled
        exit_idx = j_target
        exit_reason = None
        
        if exit_rules:
            position = Position(
                entity_id=ticker,
                entry_price=entry_px_eff,
                entry_ts=entry_ts.to_pydatetime(),
                side="LONG",
            )
            
            # Check each day for exit
            for day_idx in range(i + 1, min(i + max_hold_days + 1, len(idx))):
                current_price = float(px.iloc[day_idx]["close"])
                exit_reason = exit_rules.check_exit(position, current_price)
                
                if exit_reason:
                    exit_idx = day_idx
                    break
            
            # If no early exit, use target exit
            if not exit_reason:
                exit_idx = j_target
        
        exit_px = float(px.iloc[exit_idx]["close"])
        exit_px_eff = _apply_cost(exit_px, cost_bps, "LONG", False)

        ret = (exit_px_eff / entry_px_eff) - 1.0
        weight = 1.0 / config.max_positions
        pnl_pct = weight * ret
        equity *= (1.0 + pnl_pct)

        trades.append(
            Trade(
                entity_id=ticker,
                entry_ts=entry_ts.to_pydatetime(),
                exit_ts=idx[exit_idx].to_pydatetime(),
                side="LONG",
                qty=weight,
                entry_px=entry_px_eff,
                exit_px=exit_px_eff,
                cost_bps=cost_bps,
                pnl=pnl_pct,
                pnl_pct=pnl_pct,
            )
        )
        curve.append({"date": idx[exit_idx], "equity": equity})

    if not curve:
        return pd.DataFrame(columns=["date", "equity"]), trades
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

