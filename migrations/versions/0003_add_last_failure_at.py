"""0003 add last_failure_at to orchestrator_state

Revision ID: 0003
Revises: 0002
Create Date: 2026-01-29
"""

from __future__ import annotations

from alembic import op


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE orchestrator_state
        ADD COLUMN IF NOT EXISTS last_failure_at TIMESTAMPTZ;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE orchestrator_state
        DROP COLUMN IF EXISTS last_failure_at;
        """
    )

