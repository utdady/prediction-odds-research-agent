# SPEC_DETAILS (clarifications used by implementation)

## 1) Market ↔ Entity mapping (M:N)
`market_entity_map` allows multiple entities per market and multiple markets per entity.

Fields:
- `relationship_type`: direct | correlated | sector
- `confidence`: float 0..1
- `notes`: free text

Mapping is configured in `config/market_entity_map.yml` (human-in-loop) and can be assisted by scripts.

## 2) Timestamp alignment (anti-leakage)

- Ticks are stored at their venue timestamp `tick_ts`.
- Features are computed on a fixed snapshot grid (configurable) and **use only ticks with `tick_ts <= feature_ts`**.
- Trades execute at **next market open** after `signal_ts`.

For mock mode, we simplify:
- snapshot grid: every 6 hours UTC
- trade execution uses next business day open from local price CSV index

## 3) Probability normalization

- Use **mid** when bid/ask present: `(bid + ask)/2`.
- Use **Yes** side as canonical when available.
- If Yes/No both present, validate `abs((yes + no) - 1) <= 0.02`; else mark illiquid.
- Clip normalized probability to `[0.001, 0.999]`.
- Store both raw and normalized fields.

## 4) Walk-forward

- Rolling window: 12 months train, 1 month test
- Retrain frequency: monthly
- Purge gap: 5 trading days
- Embargo: do not trade events resolving within 7 days

## 5) Target variable

Binary classification:
- label = 1 if `log(stock_t+h / stock_t) - log(spy_t+h / spy_t) > 0`, else 0
- horizon `h` defaults to 5 trading days

## 6) Position sizing (v1)

- long-only
- max positions K=10
- equal weight 1/K for active positions
- exit: fixed holding period (default 5 trading days)

## 7) Data quality checks

Basic validators run during pipelines and write to `data_quality_log`.
If critical checks fail, orchestrator halts downstream steps.

