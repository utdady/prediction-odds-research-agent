from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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


async def mark_component_complete(component: str) -> None:
    """Mark a component as successfully completed."""
    async with get_session() as session:
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
                "last_run_at": datetime.now(timezone.utc),
                "is_dirty": False,
            },
        )
        await session.commit()


async def mark_component_failed(component: str) -> None:
    """Mark a component as failed."""
    async with get_session() as session:
        # Note: last_failure_at column may not exist in all schemas
        # Just mark as dirty for now
        await execute(
            session,
            """
            INSERT INTO orchestrator_state(component, is_dirty)
            VALUES (:component, :is_dirty)
            ON CONFLICT (component) DO UPDATE SET
              is_dirty=EXCLUDED.is_dirty
            """,
            {
                "component": component,
                "is_dirty": True,
            },
        )
        await session.commit()


async def run_once() -> None:
    """Run all pipelines, updating state incrementally after each."""
    from pipelines.ingest_markets import run as ingest_markets
    from pipelines.ingest_ticks import run as ingest_ticks
    from pipelines.build_features import run as build_features
    from pipelines.train_model import run as train_model
    from pipelines.run_inference import run as run_inference
    from pipelines.run_backtest import run as run_backtest
    from pipelines.run_backtest_walkforward import run_walk_forward
    from pipelines.publish_artifacts import run as publish_artifacts
    
    pipelines = [
        ("ingest_markets", ingest_markets),
        ("ingest_ticks", ingest_ticks),
        ("build_features", build_features),
        ("train_model", train_model),
        ("run_inference", run_inference),
        ("run_backtest", run_backtest),
        ("run_backtest_walkforward", run_walk_forward),
        ("publish_artifacts", publish_artifacts),
    ]
    
    critical_pipelines = {"ingest_markets", "ingest_ticks"}
    
    # Run each pipeline and update state incrementally
    for name, pipeline_func in pipelines:
        try:
            await pipeline_func()
            # Mark as complete immediately after success
            await mark_component_complete(name)
        except Exception as e:
            # Mark as failed immediately
            await mark_component_failed(name)
            if name in critical_pipelines:
                # For critical steps, fail fast
                raise


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

