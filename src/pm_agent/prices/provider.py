from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class PriceProvider(ABC):
    @abstractmethod
    def load_prices(self, ticker: str) -> pd.DataFrame:
        """Return DF indexed by date with columns: open, close"""
        raise NotImplementedError


class LocalCSVPriceProvider(PriceProvider):
    def __init__(self, root: str = "data/prices"):
        self.root = root

    def load_prices(self, ticker: str) -> pd.DataFrame:
        from pathlib import Path
        p = Path(f"{self.root}/{ticker}.csv")
        df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
        df = df.set_index("date")
        return df[["open", "close"]]

