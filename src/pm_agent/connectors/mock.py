from __future__ import annotations

import json
from pathlib import Path

from pm_agent.schemas import NormalizedMarket, NormalizedTick


def _read_json(path: str) -> list[dict]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


class MockConnector:
    def __init__(self, venue_id: str, markets_path: str, ticks_path: str):
        self.venue_id = venue_id
        self.markets_path = markets_path
        self.ticks_path = ticks_path

    async def fetch_markets(self) -> list[NormalizedMarket]:
        rows = _read_json(self.markets_path)
        out: list[NormalizedMarket] = []
        for r in rows:
            from datetime import datetime
            resolution_ts = None
            if r.get("resolution_ts"):
                resolution_ts = datetime.fromisoformat(r["resolution_ts"].replace("Z", "+00:00"))
            out.append(
                NormalizedMarket(
                    venue_id=self.venue_id,
                    market_id=r["market_id"],
                    event_id=r.get("event_id"),
                    title=r["title"],
                    description=r.get("description"),
                    status=r.get("status", "active"),
                    resolution_ts=resolution_ts,
                    raw=r,
                )
            )
        return out

    async def fetch_ticks(self, market_ids: list[str]) -> list[NormalizedTick]:
        rows = _read_json(self.ticks_path)
        out: list[NormalizedTick] = []
        for r in rows:
            if r["market_id"] not in market_ids:
                continue
            from datetime import datetime
            tick_ts = datetime.fromisoformat(r["tick_ts"].replace("Z", "+00:00"))
            out.append(
                NormalizedTick(
                    venue_id=self.venue_id,
                    market_id=r["market_id"],
                    tick_ts=tick_ts,
                    yes_bid=r.get("yes_bid"),
                    yes_ask=r.get("yes_ask"),
                    no_bid=r.get("no_bid"),
                    no_ask=r.get("no_ask"),
                    yes_mid=r.get("yes_mid"),
                    no_mid=r.get("no_mid"),
                    p_norm=r["p_norm"],
                    volume=r.get("volume"),
                    raw=r,
                )
            )
        return out

