"""Add theme_metrics.updated_at so same-day refreshes move the themes bootstrap revision (#526)."""

import sqlalchemy as sa

from alembic import op

revision = "20261008_0062"
down_revision = "20260928_0061"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "theme_metrics",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
    )
    # Serves max(updated_at) per pipeline on every themes bootstrap read.
    op.create_index(
        "idx_theme_metrics_pipeline_updated",
        "theme_metrics",
        ["pipeline", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_theme_metrics_pipeline_updated", table_name="theme_metrics")
    op.drop_column("theme_metrics", "updated_at")
