"""Add field-level provenance and completeness score to stock_fundamentals.

Adds two nullable columns to support T2 (quality metadata) from the ASIA
fundamentals epic:

- ``field_completeness_score`` (INT 0-100): market-aware coverage score
  computed by ``fundamentals_completeness.compute_completeness_score``.
- ``field_provenance`` (JSON): mapping of ``{field_name: provider_name}``
  for every populated field, derived per the routing policy.

Both columns are nullable with no server default — existing rows will
report ``NULL`` until the next fundamentals refresh populates them.
Consumers (scanners, API, UI) must handle ``NULL`` as "unknown quality".
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260412_0007"
down_revision = "20260412_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "stock_fundamentals",
        sa.Column("field_completeness_score", sa.Integer(), nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("field_provenance", sa.JSON(), nullable=True),
    )
    # Index for quality-aware filtering (T4 will use this to exclude low-
    # coverage rows from ranking tiers without a full table scan).
    op.create_index(
        "ix_stock_fundamentals_field_completeness_score",
        "stock_fundamentals",
        ["field_completeness_score"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_stock_fundamentals_field_completeness_score",
        table_name="stock_fundamentals",
    )
    op.drop_column("stock_fundamentals", "field_provenance")
    op.drop_column("stock_fundamentals", "field_completeness_score")
