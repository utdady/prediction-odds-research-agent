"""Exit rules for backtesting (stop-loss, take-profit, trailing stops)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """Represents an open position."""
    entity_id: str
    entry_price: float
    entry_ts: datetime
    side: str
    peak_price: float | None = None


class ExitRuleManager:
    """Manages exit rules for positions."""

    def __init__(
        self,
        stop_loss_pct: float = -0.05,
        take_profit_pct: float = 0.10,
        trailing_stop_pct: float = -0.03,
    ):
        self.stop_loss = stop_loss_pct
        self.take_profit = take_profit_pct
        self.trailing_stop = trailing_stop_pct

    def check_exit(self, position: Position, current_price: float) -> str | None:
        """Check if position should be exited. Returns exit reason or None."""
        entry_price = position.entry_price
        pnl_pct = (current_price - entry_price) / entry_price

        # Update peak price for trailing stop
        if position.peak_price is None or current_price > position.peak_price:
            position.peak_price = current_price

        # Stop loss
        if pnl_pct <= self.stop_loss:
            return "stop_loss"

        # Take profit
        if pnl_pct >= self.take_profit:
            return "take_profit"

        # Trailing stop: exit if price drops from peak
        if position.peak_price:
            drawdown = (current_price - position.peak_price) / position.peak_price
            if drawdown <= self.trailing_stop:
                return "trailing_stop"

        return None  # Hold

