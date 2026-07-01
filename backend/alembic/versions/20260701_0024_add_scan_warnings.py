"""Add scan warnings."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260701_0024"
down_revision = "20260627_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "scans",
        sa.Column(
            "warnings",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("scans", "warnings")
