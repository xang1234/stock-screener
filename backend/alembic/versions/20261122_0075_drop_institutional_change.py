"""Drop the unused stock_fundamentals.institutional_change column.

Nothing writes it. The fundamentals cache copied it into a payload key that no
caller reads. Downgrade adds it back as a nullable float, with every row NULL.
"""

import sqlalchemy as sa

from alembic import op

revision = "20261122_0075"
down_revision = "20261103_0074"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("stock_fundamentals", "institutional_change")


def downgrade() -> None:
    op.add_column("stock_fundamentals", sa.Column("institutional_change", sa.Float(), nullable=True))
