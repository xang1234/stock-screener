"""Store how far the benchmark sits below its cycle high on market_exposure.

Additive and nullable. Rows written before this migration show the breadth
divergence as unknown until backfill_exposure reruns for their market.
"""

import sqlalchemy as sa

from alembic import op

revision = "20261123_0078"
down_revision = "20261123_0077"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "market_exposure", sa.Column("index_off_high_pct", sa.Float(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("market_exposure", "index_off_high_pct")
