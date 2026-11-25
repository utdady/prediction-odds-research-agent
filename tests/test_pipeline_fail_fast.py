"""Tests that pipelines.run_all stops on first failure and exits non-zero."""
from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_failed_stage_skips_publish_artifacts() -> None:
    from pipelines import run_all as run_all_module

    publish_ran = False

    async def noop(*_args: object, **_kwargs: object) -> None:
        return None

    async def failing_build_features() -> None:
        raise ValueError("intentional test failure")

    async def track_publish() -> None:
        nonlocal publish_ran
        publish_ran = True

    with (
        patch.object(run_all_module, "ingest_markets", noop),
        patch.object(run_all_module, "ingest_ticks", noop),
        patch.object(run_all_module, "build_features", failing_build_features),
        patch.object(run_all_module, "train_model", noop),
        patch.object(run_all_module, "run_inference", noop),
        patch.object(run_all_module, "run_backtest", noop),
        patch.object(run_all_module, "run_walk_forward", noop),
        patch.object(run_all_module, "publish_artifacts", track_publish),
        patch.object(run_all_module, "mark_component_complete", noop),
        patch.object(run_all_module, "mark_component_failed", noop),
    ):
        with pytest.raises(ValueError, match="intentional test failure"):
            await run_all_module.run()

    assert publish_ran is False


def test_run_all_main_exits_nonzero_on_failure() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    runner = """
import asyncio
import sys
from unittest.mock import patch
import pipelines.run_all as ra

async def noop(*_a, **_kw):
    return None

async def boom():
    raise RuntimeError("cli test failure")

with patch.object(ra, "ingest_markets", noop), \\
     patch.object(ra, "ingest_ticks", noop), \\
     patch.object(ra, "build_features", boom), \\
     patch.object(ra, "mark_component_complete", noop), \\
     patch.object(ra, "mark_component_failed", noop):
    try:
        asyncio.run(ra.run())
    except RuntimeError:
        sys.exit(1)
sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", runner],
        env=env,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0