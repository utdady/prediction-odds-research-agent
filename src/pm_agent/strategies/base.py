"""Base strategy interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pm_agent.schemas import Signal


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    @abstractmethod
    def generate_signal(self, features: dict) -> Signal | None:
        """Generate a trading signal from features."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name."""
        pass

