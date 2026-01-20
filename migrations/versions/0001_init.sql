-- init schema (idempotent-ish; use alembic to apply once)

CREATE TABLE IF NOT EXISTS venues (
  venue_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL UNIQUE,
  name TEXT,
  sector TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  family TEXT NOT NULL,
  title TEXT NOT NULL,
  resolution_ts TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS markets (
  market_id TEXT PRIMARY KEY,
  venue_id TEXT NOT NULL REFERENCES venues(venue_id),
  event_id TEXT REFERENCES events(event_id),
  title TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL,
  yes_token TEXT,
  no_token TEXT,
  resolution_ts TIMESTAMPTZ,
  raw JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_markets_venue ON markets(venue_id);
CREATE INDEX IF NOT EXISTS idx_markets_event ON markets(event_id);

CREATE TABLE IF NOT EXISTS market_entity_map (
  map_id BIGSERIAL PRIMARY KEY,
  market_id TEXT NOT NULL REFERENCES markets(market_id),
  entity_id TEXT NOT NULL REFERENCES entities(entity_id),
  relationship_type TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  notes TEXT,
  UNIQUE(market_id, entity_id, relationship_type)
);

CREATE TABLE IF NOT EXISTS odds_ticks (
  tick_id BIGSERIAL PRIMARY KEY,
  venue_id TEXT NOT NULL REFERENCES venues(venue_id),
  market_id TEXT NOT NULL REFERENCES markets(market_id),
  tick_ts TIMESTAMPTZ NOT NULL,
  yes_bid DOUBLE PRECISION,
  yes_ask DOUBLE PRECISION,
  yes_mid DOUBLE PRECISION,
  no_bid DOUBLE PRECISION,
  no_ask DOUBLE PRECISION,
  no_mid DOUBLE PRECISION,
  p_norm DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION,
  raw JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(venue_id, market_id, tick_ts)
);

CREATE INDEX IF NOT EXISTS idx_ticks_market_ts ON odds_ticks(market_id, tick_ts);

CREATE TABLE IF NOT EXISTS features (
  feature_id BIGSERIAL PRIMARY KEY,
  entity_id TEXT NOT NULL REFERENCES entities(entity_id),
  ts TIMESTAMPTZ NOT NULL,
  p_now DOUBLE PRECISION,
  delta_p_1h DOUBLE PRECISION,
  delta_p_1d DOUBLE PRECISION,
  rolling_std_p_1d DOUBLE PRECISION,
  liquidity_score DOUBLE PRECISION,
  venue_disagreement DOUBLE PRECISION,
  time_to_resolution_days DOUBLE PRECISION,
  raw JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(entity_id, ts)
);

CREATE TABLE IF NOT EXISTS signals (
  signal_id BIGSERIAL PRIMARY KEY,
  entity_id TEXT NOT NULL REFERENCES entities(entity_id),
  ts TIMESTAMPTZ NOT NULL,
  strategy TEXT NOT NULL,
  side TEXT NOT NULL,
  strength DOUBLE PRECISION NOT NULL,
  horizon_days INT NOT NULL,
  meta JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(entity_id, ts, strategy)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  config JSONB NOT NULL,
  model_version TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS backtest_trades (
  trade_id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES backtest_runs(run_id),
  entity_id TEXT NOT NULL REFERENCES entities(entity_id),
  entry_ts TIMESTAMPTZ NOT NULL,
  exit_ts TIMESTAMPTZ NOT NULL,
  side TEXT NOT NULL,
  qty DOUBLE PRECISION NOT NULL,
  entry_px DOUBLE PRECISION NOT NULL,
  exit_px DOUBLE PRECISION NOT NULL,
  cost_bps DOUBLE PRECISION NOT NULL,
  pnl DOUBLE PRECISION NOT NULL,
  pnl_pct DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
  run_id TEXT PRIMARY KEY REFERENCES backtest_runs(run_id),
  cagr DOUBLE PRECISION,
  sharpe DOUBLE PRECISION,
  sortino DOUBLE PRECISION,
  max_drawdown DOUBLE PRECISION,
  turnover DOUBLE PRECISION,
  hit_rate DOUBLE PRECISION,
  avg_win DOUBLE PRECISION,
  avg_loss DOUBLE PRECISION,
  brier DOUBLE PRECISION,
  meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS data_quality_log (
  dq_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  scope TEXT NOT NULL,
  level TEXT NOT NULL,
  message TEXT NOT NULL,
  context JSONB NOT NULL DEFAULT '{}'::jsonb
);

