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


def downgrade() -> None:
    op.drop_column("theme_metrics", "updated_at")
