from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from pm_agent.db import get_session
from pm_agent.sql import fetch_all, fetch_one


app = FastAPI(title="pm-odds-research-agent")

REQS = Counter("api_requests_total", "Total API requests", ["path"])
LAT = Histogram("api_request_seconds", "API request latency", ["path"])


@app.get("/health")
async def health() -> dict:
    REQS.labels("/health").inc()
    return {"ok": True}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/markets/latest")
async def markets_latest(limit: int = 50) -> list[dict]:
    REQS.labels("/markets/latest").inc()
    async with get_session() as session:
        return await fetch_all(session, "SELECT * FROM markets ORDER BY updated_at DESC LIMIT :limit", {"limit": limit})


@app.get("/signals/latest")
async def signals_latest(limit: int = 50) -> list[dict]:
    REQS.labels("/signals/latest").inc()
    async with get_session() as session:
        return await fetch_all(session, "SELECT * FROM signals ORDER BY ts DESC LIMIT :limit", {"limit": limit})


@app.get("/backtests/runs")
async def backtests_runs() -> list[dict]:
    REQS.labels("/backtests/runs").inc()
    async with get_session() as session:
        return await fetch_all(session, "SELECT r.*, m.sharpe, m.max_drawdown FROM backtest_runs r LEFT JOIN backtest_metrics m ON r.run_id=m.run_id ORDER BY created_at DESC")


@app.get("/backtests/runs/{run_id}")
async def backtests_run(run_id: str) -> dict:
    REQS.labels("/backtests/runs/{run_id}").inc()
    async with get_session() as session:
        run = await fetch_one(session, "SELECT * FROM backtest_runs WHERE run_id=:run_id", {"run_id": run_id})
        trades = await fetch_all(session, "SELECT * FROM backtest_trades WHERE run_id=:run_id ORDER BY entry_ts", {"run_id": run_id})
        metrics = await fetch_one(session, "SELECT * FROM backtest_metrics WHERE run_id=:run_id", {"run_id": run_id})
    return {"run": run, "metrics": metrics, "trades": trades}


@app.get("/diagnostics/calibration")
async def diagnostics_calibration() -> dict:
    # minimal: return latest backtest metrics brier if present
    REQS.labels("/diagnostics/calibration").inc()
    async with get_session() as session:
        m = await fetch_one(session, "SELECT * FROM backtest_metrics ORDER BY run_id DESC LIMIT 1")
    return {"brier": m.get("brier") if m else None}

