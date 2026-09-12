"""Keep source-specific developments separate from theme identity."""

import sqlalchemy as sa
from alembic import op

revision = "20260910_0038"
down_revision = "20260907_0037"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("theme_mentions", sa.Column("development", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("theme_mentions") as batch:
        batch.drop_column("development")
