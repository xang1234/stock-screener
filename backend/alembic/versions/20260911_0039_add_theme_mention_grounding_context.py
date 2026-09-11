"""Persist the immutable grounding packet used for a theme mention."""

import sqlalchemy as sa
from alembic import op

revision = "20260911_0039"
down_revision = "20260910_0038"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("theme_mentions", sa.Column("grounding_context", sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table("theme_mentions") as batch:
        batch.drop_column("grounding_context")
