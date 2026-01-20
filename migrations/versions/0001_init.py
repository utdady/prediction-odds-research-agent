"""0001 init

Revision ID: 0001
Revises:
Create Date: 2026-01-19
"""

from __future__ import annotations

from alembic import op
from pathlib import Path


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = Path(__file__).parent / "0001_init.sql"
    sql = sql_path.read_text(encoding="utf-8")
    op.execute(sql)


def downgrade() -> None:
    # destructive downgrade omitted for portfolio project
    pass

