from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pipelines.run_all import run as run_all


STATE_PATH = Path("state.json")


@dataclass
class State:
    last_run: str | None = None
    dirty_flags: dict = field(default_factory=lambda: {"markets": True, "ticks": True, "features": True, "model": True, "backtest": True})


def load_state() -> State:
    if not STATE_PATH.exists():
        return State()
    return State(**json.loads(STATE_PATH.read_text(encoding="utf-8")))


def save_state(state: State) -> None:
    STATE_PATH.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


async def run_once() -> None:
    state = load_state()
    await run_all()
    state.last_run = datetime.now(timezone.utc).isoformat()
    state.dirty_flags = {k: False for k in state.dirty_flags}
    save_state(state)


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

