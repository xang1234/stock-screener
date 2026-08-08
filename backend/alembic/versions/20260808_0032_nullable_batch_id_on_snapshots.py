"""Make batch_id nullable on max_pain_snapshots and gex_snapshots.

Both tables were created with batch_id as a NOT NULL foreign key (see
20260730_0026 / 20260730_0027), on the assumption every snapshot row belongs
to a universe-wide batch run. The per-expiration term-structure endpoint
(GET /v1/options/term-structure/{symbol}) upserts these same tables for a
single ticker+expiration outside of any batch run and has no batch_id to
supply, so those inserts violate the NOT NULL constraint. The ORM models
(app/models/max_pain.py, app/models/gex.py) already declare batch_id as
nullable=True -- this migration brings the actual schema in line with that.
The FK constraint itself is left in place; NULL is always valid for a
nullable FK column.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260808_0032"
down_revision = "20260806_0031"
branch_labels = None
depends_on = None

_TABLES = ("max_pain_snapshots", "gex_snapshots")


def upgrade() -> None:
    for table in _TABLES:
        op.alter_column(table, "batch_id", existing_type=sa.String(50), nullable=True)


def downgrade() -> None:
    for table in _TABLES:
        op.alter_column(table, "batch_id", existing_type=sa.String(50), nullable=False)
