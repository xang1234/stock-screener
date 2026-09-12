"""Retain held candidates and support status independently of accepted mentions."""

import sqlalchemy as sa
from alembic import op

revision = "20260911_0040"
down_revision = "20260911_0039"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "content_item_pipeline_state",
        sa.Column("claim_review", sa.JSON(), nullable=True),
    )
    op.add_column(
        "theme_mentions", sa.Column("claim_support", sa.JSON(), nullable=True)
    )


def downgrade():
    with op.batch_alter_table("theme_mentions") as batch:
        batch.drop_column("claim_support")
    with op.batch_alter_table("content_item_pipeline_state") as batch:
        batch.drop_column("claim_review")
