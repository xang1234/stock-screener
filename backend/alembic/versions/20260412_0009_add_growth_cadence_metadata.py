"""Add growth cadence metadata columns to stock_fundamentals.

T7 adds cadence-aware growth semantics for mixed reporting frequencies.
These columns persist the emitted metadata so API/cache reads remain
deterministic after Redis expiry and across process restarts.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260412_0009"
down_revision = "20260412_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "stock_fundamentals",
        sa.Column("growth_reporting_cadence", sa.String(length=24), nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("growth_metric_basis", sa.String(length=40), nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("growth_comparable_period_date", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("growth_reference_gap_days", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("stock_fundamentals", "growth_reference_gap_days")
    op.drop_column("stock_fundamentals", "growth_comparable_period_date")
    op.drop_column("stock_fundamentals", "growth_metric_basis")
    op.drop_column("stock_fundamentals", "growth_reporting_cadence")
