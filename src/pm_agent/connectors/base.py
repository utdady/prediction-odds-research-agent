from __future__ import annotations

from abc import ABC, abstractmethod

from pm_agent.schemas import NormalizedMarket, NormalizedTick


class Connector(ABC):
    @abstractmethod
    async def fetch_markets(self) -> list[NormalizedMarket]:
        raise NotImplementedError

    @abstractmethod
    async def fetch_ticks(self, market_ids: list[str]) -> list[NormalizedTick]:
        raise NotImplementedError

