from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class NormalizedMarket(BaseModel):
    venue_id: str
    market_id: str
    event_id: str | None = None
    title: str
    description: str | None = None
    status: str
    resolution_ts: datetime | None = None
    raw: dict


class NormalizedTick(BaseModel):
    venue_id: str
    market_id: str
    tick_ts: datetime

    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None

    yes_mid: float | None = None
    no_mid: float | None = None

    p_norm: float = Field(..., ge=0.0, le=1.0)
    volume: float | None = Field(default=None, ge=0.0)
    raw: dict


class FeatureRow(BaseModel):
    entity_id: str
    ts: datetime

    p_now: float | None = None
    delta_p_1h: float | None = None
    delta_p_1d: float | None = None
    rolling_std_p_1d: float | None = None
    liquidity_score: float | None = None
    venue_disagreement: float | None = None
    time_to_resolution_days: float | None = None
    raw: dict = Field(default_factory=dict)


class Signal(BaseModel):
    entity_id: str
    ts: datetime
    strategy: str
    side: str
    strength: float
    horizon_days: int
    meta: dict = Field(default_factory=dict)

