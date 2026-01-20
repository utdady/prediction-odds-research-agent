from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pipelines.run_all import run as run_all
from pm_agent.db import get_session
from pm_agent.sql import execute, fetch_all


@dataclass
class State:
    last_run: str | None = None
    dirty_flags: dict = field(
        default_factory=lambda: {
            "markets": True,
            "ticks": True,
            "features": True,
            "model": True,
            "backtest": True,
        }
    )


async def load_state() -> State:
    async with get_session() as session:
        rows = await fetch_all(
            session,
            "SELECT component, last_run_at, is_dirty FROM orchestrator_state",
        )
    if not rows:
        return State()
    dirty = {r["component"]: r["is_dirty"] for r in rows}
    last_run = max((r["last_run_at"] for r in rows if r["last_run_at"]), default=None)
    return State(last_run=last_run.isoformat() if last_run else None, dirty_flags=dirty)


async def save_state(state: State) -> None:
    async with get_session() as session:
        for component, is_dirty in state.dirty_flags.items():
            await execute(
                session,
                """
                INSERT INTO orchestrator_state(component, last_run_at, is_dirty)
                VALUES (:component, :last_run_at, :is_dirty)
                ON CONFLICT (component) DO UPDATE SET
                  last_run_at=EXCLUDED.last_run_at,
                  is_dirty=EXCLUDED.is_dirty
                """,
                {
                    "component": component,
                    "last_run_at": datetime.now(timezone.utc)
                    if not is_dirty
                    else None,
                    "is_dirty": is_dirty,
                },
            )
        await session.commit()


async def run_once() -> None:
    state = await load_state()
    await run_all()
    state.last_run = datetime.now(timezone.utc).isoformat()
    state.dirty_flags = {k: False for k in state.dirty_flags}
    await save_state(state)


async def run_scheduled(interval_sec: int) -> None:
    while True:
        await run_once()
        await asyncio.sleep(interval_sec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["once", "scheduled"], default="once")
    ap.add_argument("--interval-sec", type=int, default=3600)
    args = ap.parse_args()

    if args.mode == "once":
        asyncio.run(run_once())
    else:
        asyncio.run(run_scheduled(args.interval_sec))


if __name__ == "__main__":
    main()

