"""Run the test suite against an isolated, freshly migrated database.

The dev database (``pm_research``) is never touched: tests use ``pm_research_test``,
which is dropped and recreated at the start of every session. Override the target with
``TEST_DATABASE_URL_SYNC`` (a SQLAlchemy ``postgresql+psycopg://`` URL); the connecting
role needs CREATEDB.

``pm_agent.config.settings`` is built at import time, so the env vars must be set in
``pytest_configure``, before any test module imports ``pm_agent``.
"""
from __future__ import annotations

import os
from pathlib import Path

import sqlalchemy
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_URL = "postgresql+psycopg://pm:pm@localhost:5432/pm_research_test"


def _recreate_database(url: sqlalchemy.engine.URL) -> None:
    admin = sqlalchemy.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    quoted = admin.dialect.identifier_preparer.quote(url.database)
    with admin.connect() as conn:
        conn.execute(sqlalchemy.text(f"DROP DATABASE IF EXISTS {quoted} WITH (FORCE)"))
        conn.execute(sqlalchemy.text(f"CREATE DATABASE {quoted}"))
    admin.dispose()


def pytest_configure(config) -> None:
    sync_url = make_url(os.getenv("TEST_DATABASE_URL_SYNC", DEFAULT_TEST_URL))
    assert sync_url.database and sync_url.database.endswith("_test"), (
        f"Refusing to run tests against non-test database {sync_url.database!r}"
    )
    os.environ["DATABASE_URL_SYNC"] = sync_url.render_as_string(hide_password=False)
    os.environ["DATABASE_URL_ASYNC"] = sync_url.set(drivername="postgresql+asyncpg").render_as_string(
        hide_password=False
    )

    _recreate_database(sync_url)

    alembic_cfg = Config(str(ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(ROOT / "migrations"))
    command.upgrade(alembic_cfg, "head")
