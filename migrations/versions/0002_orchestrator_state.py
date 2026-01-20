"""0002 add orchestrator_state table

Revision ID: 0002
Revises: 0001
Create Date: 2026-01-20
"""

from __future__ import annotations

from alembic import op


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS orchestrator_state (
          component TEXT PRIMARY KEY,
          last_run_at TIMESTAMPTZ,
          is_dirty BOOLEAN DEFAULT false,
          metadata JSONB DEFAULT '{}'::jsonb
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS orchestrator_state;")


