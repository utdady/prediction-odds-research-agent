from __future__ import annotations

import asyncio
from pathlib import Path

from pm_agent.logging import configure_logging
import structlog


log = structlog.get_logger(__name__)


async def run() -> None:
    configure_logging()
    Path("artifacts").mkdir(exist_ok=True)
    # placeholder: in real life, upload to S3/GCS
    log.info("publish_artifacts_done", path="artifacts/")


if __name__ == "__main__":
    asyncio.run(run())

