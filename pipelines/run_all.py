from __future__ import annotations

import asyncio

from pipelines.ingest_markets import run as ingest_markets
from pipelines.ingest_ticks import run as ingest_ticks
from pipelines.build_features import run as build_features
from pipelines.train_model import run as train_model
from pipelines.run_inference import run as run_inference
from pipelines.run_backtest import run as run_backtest
from pipelines.run_backtest_walkforward import run_walk_forward
from pipelines.publish_artifacts import run as publish_artifacts


async def run() -> None:
    await ingest_markets()
    await ingest_ticks()
    await build_features()
    await train_model()
    await run_inference()
    # quick single-pass backtest
    await run_backtest()
    # more realistic walk-forward run
    await run_walk_forward()
    await publish_artifacts()


if __name__ == "__main__":
    asyncio.run(run())

