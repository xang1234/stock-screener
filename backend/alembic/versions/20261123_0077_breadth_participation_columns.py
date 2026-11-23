"""Store breadth participation on market_breadth.

Additive and nullable. The calculation revision is unchanged, so rows written
before this migration stay readable and show these fields as missing until the
breadth rebuild refills history.
"""

import sqlalchemy as sa

from alembic import op

revision = "20261123_0077"
down_revision = "20261122_0076"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("above_sma200_count", sa.Integer()),
    ("above_sma200_pct", sa.Float()),
    ("down_20pct_from_high_count", sa.Integer()),
    ("down_20pct_from_high_pct", sa.Float()),
    ("up_4pct_near_high_count", sa.Integer()),
    ("sma200_eligible_count", sa.Integer()),
    ("off_high_eligible_count", sa.Integer()),
)


def upgrade() -> None:
    for name, type_ in _COLUMNS:
        op.add_column("market_breadth", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_COLUMNS):
        op.drop_column("market_breadth", name)
