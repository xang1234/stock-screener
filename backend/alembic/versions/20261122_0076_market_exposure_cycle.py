"""Store the market cycle on market_exposure.

Additive. Existing rows stay NULL until backfill_exposure reruns for their
market, which also rewrites follow_through_day and follow_through_date under
the state machine's stricter meaning.
"""

import sqlalchemy as sa

from alembic import op

revision = "20261122_0076"
down_revision = "20261122_0075"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("cycle_state", sa.String(16)),
    ("rally_day", sa.Integer()),
    ("rally_low_date", sa.Date()),
    ("failed_ftd_date", sa.Date()),
)


def upgrade() -> None:
    for name, type_ in _COLUMNS:
        op.add_column("market_exposure", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_COLUMNS):
        op.drop_column("market_exposure", name)
