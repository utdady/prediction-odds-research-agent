"""Regression tests for the security hardening changes (API limit bounds, parameterized SQL)."""
from __future__ import annotations

import pytest
import sqlalchemy
from fastapi.testclient import TestClient


@pytest.mark.parametrize("path", ["/markets/latest", "/signals/latest"])
@pytest.mark.parametrize("limit", [0, -1, 501])
def test_api_limit_out_of_range_is_rejected(path: str, limit: int) -> None:
    from app.api.main import app

    # FastAPI validates query params before the handler runs, so no DB is needed.
    resp = TestClient(app).get(path, params={"limit": limit})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "query",
    [
        "company AAPL",
        "signals for AAPL",
        "recent signals",
        "markets for AAPL",
        "recent markets",
        "features for AAPL",
        "recent features",
    ],
)
def test_search_agent_queries_execute(query: str) -> None:
    """Every SearchAgent SQL path (now parameterized) must run without error."""
    from pm_agent.agent.search_agent import SearchAgent
    from pm_agent.config import settings

    engine = sqlalchemy.create_engine(settings.database_url_sync)
    try:
        result = SearchAgent(engine).search(query)
    finally:
        engine.dispose()
    assert "answer" in result
