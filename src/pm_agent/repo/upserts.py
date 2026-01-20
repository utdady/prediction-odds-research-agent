from __future__ import annotations

import json
from typing import Iterable

from sqlalchemy.ext.asyncio import AsyncSession

from pm_agent.schemas import FeatureRow, NormalizedMarket, NormalizedTick, Signal
from pm_agent.sql import execute


async def upsert_event(
    session: AsyncSession,
    *,
    event_id: str,
    family: str,
    title: str,
    resolution_ts,
) -> None:
    await execute(
        session,
        """
        INSERT INTO events(event_id, family, title, resolution_ts)
        VALUES (:event_id, :family, :title, :resolution_ts)
        ON CONFLICT (event_id) DO UPDATE SET
          family=EXCLUDED.family,
          title=EXCLUDED.title,
          resolution_ts=EXCLUDED.resolution_ts
        """,
        {
            "event_id": event_id,
            "family": family,
            "title": title,
            "resolution_ts": resolution_ts,
        },
    )


async def upsert_venue(session: AsyncSession, venue_id: str, name: str) -> None:
    await execute(
        session,
        """
        INSERT INTO venues(venue_id, name)
        VALUES (:venue_id, :name)
        ON CONFLICT (venue_id) DO UPDATE SET name=EXCLUDED.name
        """,
        {"venue_id": venue_id, "name": name},
    )


async def upsert_market(session: AsyncSession, m: NormalizedMarket) -> None:
    await execute(
        session,
        """
        INSERT INTO markets(market_id, venue_id, event_id, title, description, status, resolution_ts, raw, updated_at)
        VALUES (:market_id, :venue_id, :event_id, :title, :description, :status, :resolution_ts, CAST(:raw AS jsonb), NOW())
        ON CONFLICT (market_id) DO UPDATE SET
          event_id=EXCLUDED.event_id,
          title=EXCLUDED.title,
          description=EXCLUDED.description,
          status=EXCLUDED.status,
          resolution_ts=EXCLUDED.resolution_ts,
          raw=EXCLUDED.raw,
          updated_at=NOW()
        """,
        {
            "market_id": m.market_id,
            "venue_id": m.venue_id,
            "event_id": m.event_id,
            "title": m.title,
            "description": m.description,
            "status": m.status,
            "resolution_ts": m.resolution_ts,
            "raw": json.dumps(m.raw),
        },
    )


async def upsert_tick(session: AsyncSession, t: NormalizedTick) -> None:
    await execute(
        session,
        """
        INSERT INTO odds_ticks(
          venue_id, market_id, tick_ts,
          yes_bid, yes_ask, yes_mid,
          no_bid, no_ask, no_mid,
          p_norm, volume, raw
        )
        VALUES (
          :venue_id, :market_id, :tick_ts,
          :yes_bid, :yes_ask, :yes_mid,
          :no_bid, :no_ask, :no_mid,
          :p_norm, :volume, CAST(:raw AS jsonb)
        )
        ON CONFLICT (venue_id, market_id, tick_ts) DO UPDATE SET
          yes_bid=EXCLUDED.yes_bid,
          yes_ask=EXCLUDED.yes_ask,
          yes_mid=EXCLUDED.yes_mid,
          no_bid=EXCLUDED.no_bid,
          no_ask=EXCLUDED.no_ask,
          no_mid=EXCLUDED.no_mid,
          p_norm=EXCLUDED.p_norm,
          volume=EXCLUDED.volume,
          raw=EXCLUDED.raw
        """,
        {
            "venue_id": t.venue_id,
            "market_id": t.market_id,
            "tick_ts": t.tick_ts,
            "yes_bid": t.yes_bid,
            "yes_ask": t.yes_ask,
            "yes_mid": t.yes_mid,
            "no_bid": t.no_bid,
            "no_ask": t.no_ask,
            "no_mid": t.no_mid,
            "p_norm": t.p_norm,
            "volume": t.volume,
            "raw": json.dumps(t.raw),
        },
    )


async def upsert_features(session: AsyncSession, rows: Iterable[FeatureRow]) -> None:
    for r in rows:
        await execute(
            session,
            """
            INSERT INTO features(entity_id, ts, p_now, delta_p_1h, delta_p_1d, rolling_std_p_1d,
                                 liquidity_score, venue_disagreement, time_to_resolution_days, raw)
            VALUES (:entity_id, :ts, :p_now, :delta_p_1h, :delta_p_1d, :rolling_std_p_1d,
                    :liquidity_score, :venue_disagreement, :time_to_resolution_days, CAST(:raw AS jsonb))
            ON CONFLICT (entity_id, ts) DO UPDATE SET
              p_now=EXCLUDED.p_now,
              delta_p_1h=EXCLUDED.delta_p_1h,
              delta_p_1d=EXCLUDED.delta_p_1d,
              rolling_std_p_1d=EXCLUDED.rolling_std_p_1d,
              liquidity_score=EXCLUDED.liquidity_score,
              venue_disagreement=EXCLUDED.venue_disagreement,
              time_to_resolution_days=EXCLUDED.time_to_resolution_days,
              raw=EXCLUDED.raw
            """,
            {
                "entity_id": r.entity_id,
                "ts": r.ts,
                "p_now": r.p_now,
                "delta_p_1h": r.delta_p_1h,
                "delta_p_1d": r.delta_p_1d,
                "rolling_std_p_1d": r.rolling_std_p_1d,
                "liquidity_score": r.liquidity_score,
                "venue_disagreement": r.venue_disagreement,
                "time_to_resolution_days": r.time_to_resolution_days,
                "raw": json.dumps(r.raw),
            },
        )


async def upsert_signal(session: AsyncSession, s: Signal) -> None:
    await execute(
        session,
        """
        INSERT INTO signals(entity_id, ts, strategy, side, strength, horizon_days, meta)
        VALUES (:entity_id, :ts, :strategy, :side, :strength, :horizon_days, CAST(:meta AS jsonb))
        ON CONFLICT (entity_id, ts, strategy) DO UPDATE SET
          side=EXCLUDED.side,
          strength=EXCLUDED.strength,
          horizon_days=EXCLUDED.horizon_days,
          meta=EXCLUDED.meta
        """,
        {
            "entity_id": s.entity_id,
            "ts": s.ts,
            "strategy": s.strategy,
            "side": s.side,
            "strength": s.strength,
            "horizon_days": s.horizon_days,
            "meta": json.dumps(s.meta),
        },
    )

