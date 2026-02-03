"""
Utility to clear noisy rows from data_quality_log.

By default this deletes only `timestamp_invalid` rows which were caused by older mock data.
Safe to run locally; intended to reset the Streamlit "Data Quality Issues" view.
"""

from __future__ import annotations

import argparse

import sqlalchemy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db-url",
        default="postgresql+psycopg://pm:pm@localhost:5432/pm_research",
        help="Sync SQLAlchemy DB URL",
    )
    p.add_argument(
        "--message",
        default="timestamp_invalid",
        help="data_quality_log.message value to delete",
    )
    args = p.parse_args()

    engine = sqlalchemy.create_engine(args.db_url)
    # Use SQLAlchemy text() to support named parameters across drivers.
    stmt = sqlalchemy.text("DELETE FROM data_quality_log WHERE message = :msg")
    with engine.begin() as conn:
        res = conn.execute(stmt, {"msg": args.message})

    # SQLAlchemy may not always provide rowcount depending on driver, but psycopg does.
    print(f"Deleted rows where message='{args.message}'. rowcount={getattr(res, 'rowcount', None)}")


if __name__ == "__main__":
    main()


