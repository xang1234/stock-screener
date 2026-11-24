"""Store the 20- and 50-day shares and trend-template entries on market_breadth.

Additive and nullable. The calculation revision is unchanged, so rows written
before this migration stay readable and show these fields as missing until the
breadth rebuild refills history: ``python -m app.scripts.rebuild_market_breadth
build --start-date <date>`` for every market (validation rejects a partial
market set, and activation replaces all live rows), then ``validate`` and
``activate --confirm-replace``.
"""

import sqlalchemy as sa

from alembic import op

revision = "20261124_0079"
down_revision = "20261123_0078"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("above_sma20_count", sa.Integer()),
    ("above_sma20_pct", sa.Float()),
    ("above_sma50_count", sa.Integer()),
    ("above_sma50_pct", sa.Float()),
    ("trend_template_entry_count", sa.Integer()),
    ("sma20_eligible_count", sa.Integer()),
    ("sma50_eligible_count", sa.Integer()),
    ("trend_template_entry_eligible_count", sa.Integer()),
)


def upgrade() -> None:
    for name, type_ in _COLUMNS:
        op.add_column("market_breadth", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_COLUMNS):
        op.drop_column("market_breadth", name)
