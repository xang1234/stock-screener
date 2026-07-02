"""Merge rs_line_leadership_fields and signal_archive branches."""

from __future__ import annotations

from alembic import op

revision = "20260702_0024"
down_revision = ("20260627_0023", "20260628_0023")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
