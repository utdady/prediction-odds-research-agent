"""Paper trading engine for live simulation."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pm_agent.schemas import Signal

log = structlog.get_logger(__name__)


@dataclass
class PaperPosition:
    """Represents a paper trading position."""

    entity_id: str
    shares: float
    entry_price: float
    entry_time: datetime
    signal_strength: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PaperTradingEngine:
    """Paper trading engine for live simulation."""

    initial_capital: float = 100000.0
    cash: float = field(init=False)
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    trade_log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_capital

    async def run_live(
        self,
        fetch_live_data_func,
        compute_features_func,
        generate_signals_func,
        fetch_price_func,
        cycle_hours: int = 6,
    ) -> None:
        """
        Run strategy in real-time with live data.
        
        Args:
            fetch_live_data_func: Async function to fetch live prediction market data
            compute_features_func: Async function to compute features
            generate_signals_func: Async function to generate signals
            fetch_price_func: Async function to fetch current stock price
            cycle_hours: Hours between cycles
        """
        log.info("paper_trading_started", initial_capital=self.initial_capital)

        while True:
            try:
                # Fetch live prediction market data
                kalshi_data, poly_data = await fetch_live_data_func()

                # Ingest to database (would call actual ingestion pipeline)
                # await ingest_live_data(kalshi_data, poly_data)

                # Compute features (using live cutoff logic)
                features = await compute_features_func()

                # Generate signals
                signals = await generate_signals_func(features)

                # Execute paper trades
                for signal in signals:
                    await self.execute_paper_trade(signal, fetch_price_func)

                # Check existing positions for exits
                await self.check_exits(fetch_price_func)

                # Log performance
                await self.log_performance()

                # Sleep until next cycle
                log.info("paper_trading_cycle_complete", sleep_hours=cycle_hours)
                await asyncio.sleep(cycle_hours * 3600)

            except Exception as e:
                log.error("paper_trading_error", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def execute_paper_trade(
        self, signal: Signal, fetch_price_func
    ) -> None:
        """
        Simulate trade execution with live prices.
        
        Args:
            signal: Trading signal
            fetch_price_func: Function to fetch current price
        """
        try:
            # Get current market price (use real API)
            current_price = await fetch_price_func(signal.entity_id)

            if current_price is None or current_price <= 0:
                log.warning("invalid_price", entity_id=signal.entity_id)
                return

            # Apply realistic slippage
            if signal.side == "LONG":
                execution_price = current_price * 1.001  # 10 bps slippage
            else:
                execution_price = current_price * 0.999

            # Calculate position size (10% of capital per position)
            position_value = self.cash * 0.1
            shares = position_value / execution_price

            if shares <= 0 or position_value > self.cash:
                log.warning("insufficient_cash", entity_id=signal.entity_id)
                return

            # Record trade
            self.positions[signal.entity_id] = PaperPosition(
                entity_id=signal.entity_id,
                shares=shares,
                entry_price=execution_price,
                entry_time=datetime.now(),
                signal_strength=signal.strength,
                current_price=execution_price,
            )

            self.cash -= position_value

            self.trade_log.append(
                {
                    "timestamp": datetime.now(),
                    "entity_id": signal.entity_id,
                    "action": "BUY",
                    "shares": shares,
                    "price": execution_price,
                    "value": position_value,
                    "signal_strength": signal.strength,
                }
            )

            log.info(
                "paper_trade_executed",
                entity_id=signal.entity_id,
                shares=shares,
                price=execution_price,
                signal_strength=signal.strength,
            )

        except Exception as e:
            log.error("paper_trade_failed", entity_id=signal.entity_id, error=str(e))

    async def check_exits(self, fetch_price_func) -> None:
        """Check existing positions for exit conditions."""
        for entity_id, position in list(self.positions.items()):
            try:
                current_price = await fetch_price_func(entity_id)

                if current_price is None:
                    continue

                position.current_price = current_price
                position.unrealized_pnl = (
                    (current_price - position.entry_price) / position.entry_price
                )

                # Simple exit rule: take profit at 10% or stop loss at -5%
                if position.unrealized_pnl >= 0.10 or position.unrealized_pnl <= -0.05:
                    await self.close_position(entity_id, current_price)

            except Exception as e:
                log.warning("exit_check_failed", entity_id=entity_id, error=str(e))

    async def close_position(self, entity_id: str, exit_price: float) -> None:
        """Close a position."""
        if entity_id not in self.positions:
            return

        position = self.positions[entity_id]
        exit_value = position.shares * exit_price * 0.999  # Slippage on exit

        self.cash += exit_value

        realized_pnl = (exit_price - position.entry_price) / position.entry_price

        self.trade_log.append(
            {
                "timestamp": datetime.now(),
                "entity_id": entity_id,
                "action": "SELL",
                "shares": position.shares,
                "price": exit_price,
                "value": exit_value,
                "realized_pnl": realized_pnl,
            }
        )

        log.info(
            "paper_position_closed",
            entity_id=entity_id,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl=realized_pnl,
        )

        del self.positions[entity_id]

    async def log_performance(self) -> None:
        """Log current performance metrics."""
        total_value = self.cash

        for position in self.positions.values():
            total_value += position.shares * position.current_price

        total_return = (total_value / self.initial_capital) - 1.0

        log.info(
            "paper_trading_performance",
            cash=self.cash,
            positions=len(self.positions),
            total_value=total_value,
            total_return=total_return,
        )

