"""Record which taxonomy version the legacy_mapping development links are for (#513)."""

import sqlalchemy as sa

from alembic import op

revision = "20261010_0063"
down_revision = "20261008_0062"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "economic_development_backfills",
        sa.Column("taxonomy_version_id", sa.Uuid(), primary_key=True),
        sa.Column("through_observation_id", sa.Integer(), nullable=False),
        sa.Column("legacy_link_count", sa.Integer(), nullable=False),
        sa.Column("legacy_max_observation_id", sa.Integer(), nullable=True),
        sa.Column(
            "unallocated_observation_ids",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["taxonomy_version_id"],
            ["economic_taxonomy_versions.id"],
            ondelete="CASCADE",
        ),
    )


def downgrade() -> None:
    # The earlier builder maps legacy links itself; backfilled rows would
    # duplicate or outlive them.
    op.execute(
        "DELETE FROM economic_theme_developments WHERE link_origin = 'legacy_mapping'"
    )
    op.drop_table("economic_development_backfills")
