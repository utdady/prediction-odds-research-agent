from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "pm-odds-research-agent"

    database_url_async: str = "postgresql+asyncpg://pm:pm@localhost:5432/pm_research"
    database_url_sync: str = "postgresql+psycopg://pm:pm@localhost:5432/pm_research"

    mock_mode: bool = True

    mlflow_tracking_uri: str | None = "http://localhost:5000"
    mlflow_experiment: str = "pm_odds_signals"

    snapshot_hours: int = 6
    feature_family: str = "company_event"

    rule_delta_p_1d_threshold: float = 0.08
    rule_min_liquidity: float = 0.2
    holding_period_days: int = 5

    max_positions: int = 10
    cost_spread_bps: float = 5.0
    cost_slippage_bps: float = 5.0

    walk_train_days: int = 252
    walk_test_days: int = 21
    walk_purge_days: int = 5

    ml_confidence_threshold: float = 0.6


settings = Settings()

