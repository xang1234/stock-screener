"""Retain every observing source for canonical content eligibility."""

from alembic import op
import sqlalchemy as sa


revision = "20260909_0039"
down_revision = "20260908_0038"
branch_labels = None
depends_on = None

TABLE = "content_pipeline_eligibility"
UPGRADE_TABLE = "content_pipeline_eligibility_source_memberships"
DOWNGRADE_TABLE = "content_pipeline_eligibility_legacy_identity"


def _source_columns():
    return (
        sa.Column("content_item_id", sa.Integer(), nullable=False),
        sa.Column("pipeline", sa.Text(), nullable=False),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("originating_source_id", sa.Integer(), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
    )


def _create_source_memberships(name):
    return op.create_table(
        name,
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            primary_key=True,
            autoincrement=True,
        ),
        *_source_columns(),
        sa.ForeignKeyConstraint(
            ["content_item_id"], ["content_items.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["originating_source_id"],
            ["content_sources.id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "content_item_id",
            "pipeline",
            "channel",
            "originating_source_id",
            name="uq_content_eligibility_source",
        ),
        sa.CheckConstraint(
            "channel IN ('legacy','social')",
            name="ck_content_eligibility_channel",
        ),
        sa.CheckConstraint(
            "pipeline IN ('technical','fundamental')",
            name="ck_content_eligibility_pipeline",
        ),
    )


def _create_legacy_identity(name):
    return op.create_table(
        name,
        *_source_columns(),
        sa.ForeignKeyConstraint(
            ["content_item_id"], ["content_items.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["originating_source_id"],
            ["content_sources.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("content_item_id", "pipeline", "channel"),
        sa.CheckConstraint(
            "channel IN ('legacy','social')",
            name="ck_content_eligibility_channel",
        ),
        sa.CheckConstraint(
            "pipeline IN ('technical','fundamental')",
            name="ck_content_eligibility_pipeline",
        ),
    )


def _old_table(name):
    return sa.table(name, *_source_columns())


def upgrade():
    target = _create_source_memberships(UPGRADE_TABLE)
    source = _old_table(TABLE)
    columns = (
        "content_item_id",
        "pipeline",
        "channel",
        "originating_source_id",
        "observed_at",
    )
    op.execute(target.insert().from_select(columns, sa.select(
        source.c.content_item_id,
        source.c.pipeline,
        source.c.channel,
        source.c.originating_source_id,
        source.c.observed_at,
    )))
    op.drop_table(TABLE)
    op.rename_table(UPGRADE_TABLE, TABLE)


def downgrade():
    target = _create_legacy_identity(DOWNGRADE_TABLE)
    source = _old_table(TABLE)
    columns = (
        "content_item_id",
        "pipeline",
        "channel",
        "originating_source_id",
        "observed_at",
    )
    op.execute(target.insert().from_select(columns, sa.select(
        source.c.content_item_id,
        source.c.pipeline,
        source.c.channel,
        sa.func.min(source.c.originating_source_id),
        sa.func.min(source.c.observed_at),
    ).group_by(
        source.c.content_item_id,
        source.c.pipeline,
        source.c.channel,
    )))
    op.drop_table(TABLE)
    op.rename_table(DOWNGRADE_TABLE, TABLE)
