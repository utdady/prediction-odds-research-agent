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
    """Comprehensive health check endpoint."""
    REQS.labels("/health").inc()
    from datetime import datetime, timezone
    from pathlib import Path
    
    checks: dict[str, str] = {}
    
    # Database check
    try:
        async with get_session() as session:
            await fetch_one(session, "SELECT 1")
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    # Check if model exists
    model_path = Path("artifacts/model_v1.joblib")
    checks["model"] = "available" if model_path.exists() else "missing"
    
    # Check data freshness
    try:
        async with get_session() as session:
            latest_feature = await fetch_one(
                session,
                "SELECT MAX(ts) as max_ts FROM features"
            )
            latest_tick = await fetch_one(
                session,
                "SELECT MAX(tick_ts) as max_ts FROM odds_ticks"
            )
            
            if latest_feature and latest_feature.get("max_ts"):
                feature_age = (datetime.now(timezone.utc) - latest_feature["max_ts"]).total_seconds() / 3600
                checks["features_age_hours"] = f"{feature_age:.1f}"
            else:
                checks["features_age_hours"] = "no_data"
                
            if latest_tick and latest_tick.get("max_ts"):
                tick_age = (datetime.now(timezone.utc) - latest_tick["max_ts"]).total_seconds() / 3600
                checks["ticks_age_hours"] = f"{tick_age:.1f}"
            else:
                checks["ticks_age_hours"] = "no_data"
    except Exception as e:
        checks["data_freshness"] = f"error: {str(e)}"
    
    # Check orchestrator state
    try:
        async with get_session() as session:
            state_rows = await fetch_all(
                session,
                "SELECT component, last_run_at, is_dirty FROM orchestrator_state"
            )
            if state_rows:
                checks["orchestrator"] = "configured"
                # Count dirty components
                dirty_count = sum(1 for r in state_rows if r.get("is_dirty"))
                checks["dirty_components"] = str(dirty_count)
            else:
                checks["orchestrator"] = "not_configured"
    except Exception as e:
        checks["orchestrator"] = f"error: {str(e)}"
    
    # Overall status
    is_healthy = (
        checks.get("database") == "healthy" and
        checks.get("features_age_hours") != "no_data"
    )
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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

