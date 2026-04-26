"""Add indexes for runtime scan readiness lookups."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260426_0017"
down_revision = "20260425_0016"
branch_labels = None
depends_on = None


def _index_names(bind) -> set[str]:
    return {index["name"] for index in sa.inspect(bind).get_indexes("scans")}


def upgrade() -> None:
    bind = op.get_bind()
    existing = _index_names(bind)
    if "ix_scans_status" not in existing:
        op.create_index("ix_scans_status", "scans", ["status"])
    if "idx_scans_market_status_trigger" not in existing:
        op.create_index(
            "idx_scans_market_status_trigger",
            "scans",
            ["universe_market", "status", "trigger_source"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    existing = _index_names(bind)
    if "idx_scans_market_status_trigger" in existing:
        op.drop_index("idx_scans_market_status_trigger", table_name="scans")
    if "ix_scans_status" in existing:
        op.drop_index("ix_scans_status", table_name="scans")
