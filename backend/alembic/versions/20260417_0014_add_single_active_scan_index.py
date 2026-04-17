"""Enforce the single-active-scan invariant at the database layer."""

from __future__ import annotations

from alembic import op

revision = "20260417_0014"
down_revision = "20260415_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect in {"postgresql", "sqlite"}:
        op.execute(
            """
            CREATE UNIQUE INDEX uq_scans_single_active
            ON scans ((CASE WHEN status IN ('queued', 'running') THEN 1 END))
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect in {"postgresql", "sqlite"}:
        op.execute("DROP INDEX IF EXISTS uq_scans_single_active")
