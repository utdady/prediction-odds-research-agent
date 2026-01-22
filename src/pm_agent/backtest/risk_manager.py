"""Risk management for position limits and correlation checks."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pm_agent.prices.provider import PriceProvider


class RiskManager:
    """Manages position limits and correlation constraints."""

    def __init__(
        self,
        max_sector_weight: float = 0.30,
        max_correlation: float = 0.7,
        price_provider: PriceProvider | None = None,
    ):
        self.max_sector_weight = max_sector_weight
        self.max_correlation = max_correlation
        self.price_provider = price_provider

    def can_add_position(
        self,
        new_ticker: str,
        current_positions: list[str],
        sector_map: dict[str, str] | None = None,
    ) -> tuple[bool, str | None]:
        """
        Check if new position can be added.
        Returns (can_add, reason_if_no).
        """
        if not current_positions:
            return True, None

        # Check 1: Sector concentration
        if sector_map:
            new_sector = sector_map.get(new_ticker, "unknown")
            sector_exposure = sum(
                1.0 / len(current_positions)
                for p in current_positions
                if sector_map.get(p, "unknown") == new_sector
            )

            if sector_exposure + 1.0 / (len(current_positions) + 1) > self.max_sector_weight:
                return False, f"sector_concentration_{new_sector}"

        # Check 2: Correlation with existing positions
        if self.price_provider:
            try:
                new_returns = self._get_returns(new_ticker, lookback=30)

                for existing in current_positions:
                    existing_returns = self._get_returns(existing, lookback=30)

                    if len(new_returns) > 1 and len(existing_returns) > 1:
                        correlation = np.corrcoef(new_returns, existing_returns)[0, 1]

                        if not np.isnan(correlation) and correlation > self.max_correlation:
                            return False, f"high_correlation_{existing}_{correlation:.2f}"
            except Exception:
                # If we can't compute correlation, allow the position
                pass

        return True, None

    def _get_returns(self, ticker: str, lookback: int = 30) -> pd.Series:
        """Get recent returns for correlation calculation."""
        if not self.price_provider:
            return pd.Series(dtype=float)

        prices = self.price_provider.load_prices(ticker)
        if len(prices) < lookback:
            return pd.Series(dtype=float)

        recent = prices.tail(lookback)
        returns = recent["close"].pct_change().dropna()
        return returns

