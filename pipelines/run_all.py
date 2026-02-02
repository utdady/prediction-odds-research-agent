from __future__ import annotations

import asyncio

import structlog

from agent.orchestrator import mark_component_complete, mark_component_failed
from pipelines.ingest_markets import run as ingest_markets
from pipelines.ingest_ticks import run as ingest_ticks
from pipelines.build_features import run as build_features
from pipelines.train_model import run as train_model
from pipelines.run_inference import run as run_inference
from pipelines.run_backtest import run as run_backtest
from pipelines.run_backtest_walkforward import run_walk_forward
from pipelines.publish_artifacts import run as publish_artifacts


log = structlog.get_logger(__name__)


async def run() -> dict[str, str]:
    """Run all pipelines with basic error handling and logging.

    Returns
    -------
    dict
        Mapping of pipeline name → status ('success' | 'failed')
    """
    pipeline_results: dict[str, str] = {}

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

    for name, pipeline_func in pipelines:
        try:
            log.info("pipeline_start", pipeline=name)
            await pipeline_func()
            pipeline_results[name] = "success"
            log.info("pipeline_done", pipeline=name, status="success")
            # Update orchestrator state
            await mark_component_complete(name)
        except Exception as e:  # pragma: no cover - defensive logging
            log.error("pipeline_failed", pipeline=name, error=str(e))
            pipeline_results[name] = "failed"
            # Update orchestrator state
            await mark_component_failed(name)
            if name in critical_pipelines:
                # For critical steps, fail fast
                raise

    return pipeline_results


if __name__ == "__main__":
    asyncio.run(run())

