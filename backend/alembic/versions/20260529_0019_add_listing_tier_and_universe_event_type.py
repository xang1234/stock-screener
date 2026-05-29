"""Add listing tier metadata and universe audit event types."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260529_0019"
down_revision = "20260503_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("stock_universe") as batch_op:
        batch_op.add_column(
            sa.Column("listing_tier", sa.String(length=64), nullable=True)
        )
        batch_op.create_index(
            "ix_stock_universe_listing_tier", ["listing_tier"], unique=False
        )
        batch_op.create_index(
            "idx_universe_market_listing_tier_active",
            ["market", "listing_tier", "is_active"],
            unique=False,
        )

    with op.batch_alter_table("stock_universe_status_events") as batch_op:
        batch_op.add_column(
            sa.Column(
                "event_type",
                sa.String(length=64),
                nullable=False,
                server_default="status_changed",
            )
        )
        batch_op.alter_column(
            "new_status",
            existing_type=sa.String(length=32),
            nullable=True,
        )
        batch_op.create_index(
            "ix_stock_universe_status_events_event_type",
            ["event_type"],
            unique=False,
        )
        batch_op.create_index(
            "idx_universe_status_events_type_created",
            ["event_type", "created_at"],
            unique=False,
        )


def downgrade() -> None:
    op.execute(
        "UPDATE stock_universe_status_events "
        "SET new_status = 'active' "
        "WHERE new_status IS NULL"
    )
    with op.batch_alter_table("stock_universe_status_events") as batch_op:
        batch_op.drop_index("idx_universe_status_events_type_created")
        batch_op.drop_index("ix_stock_universe_status_events_event_type")
        batch_op.alter_column(
            "new_status",
            existing_type=sa.String(length=32),
            nullable=False,
        )
        batch_op.drop_column("event_type")

    with op.batch_alter_table("stock_universe") as batch_op:
        batch_op.drop_index("idx_universe_market_listing_tier_active")
        batch_op.drop_index("ix_stock_universe_listing_tier")
        batch_op.drop_column("listing_tier")
