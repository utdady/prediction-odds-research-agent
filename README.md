# Prediction Odds Research Agent (Kalshi + Polymarket) → Equity Signals

Portfolio project: a **deterministic research agent** that ingests prediction-market odds from multiple venues (Kalshi + Polymarket), normalizes them into implied probabilities, maps markets to public tickers (event-driven), engineers features (probability shocks, liquidity, cross-venue disagreement), trains + calibrates a baseline model, and runs **walk-forward backtests vs SPY** with strong leakage controls.

- **No real-money execution**. No broker APIs. No trading.
- **Runs end-to-end with mock fixtures** (no secrets) and writes results to Postgres.

## Architecture

```
connectors (async httpx) ──> pipelines.ingest_* ──> Postgres
                                      │
                                      v
                            pipelines.build_features
                                      │
                      ┌───────────────┴───────────────┐
                      v                               v
            pipelines.train_model                pipelines.run_inference
                      │                               │
                      v                               v
                 MLflow (optional)                signals table
                                      │
                                      v
                             pipelines.run_backtest
                                      │
                                      v
                           artifacts/ + backtest_* tables

FastAPI serves latest markets/signals/backtests.
Streamlit dashboard visualizes markets → features → signals → equity curves.

Orchestrator (`agent/orchestrator.py`) coordinates steps deterministically and keeps a persisted `state.json` to avoid unnecessary recompute.
```

## Leakage prevention (no peek)

- **Time-based splits only**.
- **Feature as-of semantics**: each feature row at timestamp `ts` uses only ticks with `tick_ts <= ts`.
- **Signal execution rule**: signals generated at `ts` trade at **next market open** (simplified; configurable).
- Walk-forward evaluation uses rolling train/test windows with a **purge gap**.

See `SPEC_DETAILS.md` for concrete timestamp alignment, normalization rules, and walk-forward config.

## Quickstart (Docker + single entrypoint)

### Prereqs
- Docker Desktop
- Python 3.11+

### 1) Bring up services

```bash
make up
```

### 2) Install deps

```bash
make setup
```

### 3) Run migrations

```bash
make migrate
```

### 4) Run end-to-end pipeline (mock fixtures)

```bash
make run_all
```

This should:
- ingest venues/markets + ticks from `data/mock/`
- build features + generate at least one signal
- train + calibrate baseline model
- run a backtest and store metrics

### 5) Start API

```bash
make api
```

Endpoints:
- `GET /health`
- `GET /markets/latest?limit=...`
- `GET /signals/latest?limit=...`
- `GET /backtests/runs`
- `GET /backtests/runs/{run_id}`
- `GET /diagnostics/calibration`

### 6) Start dashboard

```bash
make dashboard
```

Dashboard tabs:
- Markets
- Signals
- Backtest
- Diagnostics

## Results

Example screenshots are placed in `docs/screenshots/` (placeholders if you haven't run yet).

## Limitations & next steps

- Improve liquidity models (orderbook depth, realized spread)
- Add more venues (Manifold, PredictIt), add options-implied probabilities
- Better timestamp alignment (exchange calendars, true open/close pricing)
- Add conformal uncertainty and position limits by sector/correlation

---

### Repo entrypoints

- Run everything: `python -m pipelines.run_all`
- Orchestrator: `python -m agent.orchestrator --mode once`
- API: `uvicorn src.app.api.main:app --reload`
- Dashboard: `streamlit run src.app.dashboard.Home`

