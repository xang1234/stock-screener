"""Add reversible theme grouping and development observations."""

import sqlalchemy as sa
from alembic import op

revision = "20260912_0043"
down_revision = "20260911_0042"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("theme_metrics", sa.Column("grouping_version", sa.String(64)))
    op.create_table(
        "theme_equivalence_operations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("operation_key", sa.String(120), nullable=False, unique=True),
        sa.Column(
            "source_id",
            sa.Integer,
            sa.ForeignKey("theme_clusters.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("requested_target_id", sa.Integer, nullable=False),
        sa.Column(
            "target_id",
            sa.Integer,
            sa.ForeignKey("theme_clusters.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("pipeline", sa.String(20), nullable=False),
        sa.Column("member_ids", sa.JSON, nullable=False),
        sa.Column("aliases", sa.JSON, nullable=False),
        sa.Column("actor", sa.String(120), nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column("refresh_pending", sa.Boolean, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("undone_at", sa.DateTime(timezone=True)),
        sa.Column("undone_by", sa.String(120)),
        sa.Column("undo_reason", sa.Text),
    )
    op.create_index(
        "ix_theme_equivalence_operations_pipeline",
        "theme_equivalence_operations",
        ["pipeline"],
    )
    op.create_table(
        "theme_development_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("pipeline", sa.String(20), nullable=False),
        sa.Column("event_key", sa.String(64), nullable=False),
        sa.Column("identity", sa.JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("pipeline", "event_key", name="uq_theme_event_key"),
    )
    op.create_index(
        "ix_theme_development_events_pipeline", "theme_development_events", ["pipeline"]
    )
    op.create_table(
        "theme_development_observations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "event_id",
            sa.Integer,
            sa.ForeignKey("theme_development_events.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "content_item_id",
            sa.Integer,
            sa.ForeignKey("content_items.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("pipeline", sa.String(20), nullable=False),
        sa.Column("revision", sa.String(64), nullable=False),
        sa.Column("observation_key", sa.String(64), nullable=False, unique=True),
        sa.Column("theme_ids", sa.JSON, nullable=False),
        sa.Column("facts", sa.JSON, nullable=False),
        sa.Column("citations", sa.JSON, nullable=False),
        sa.Column("classification", sa.String(30), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("superseded", sa.Boolean, nullable=False),
    )
    for col in ["event_id", "content_item_id"]:
        op.create_index(
            "ix_theme_development_observations_" + col,
            "theme_development_observations",
            [col],
        )
    op.create_table(
        "theme_development_themes",
        sa.Column(
            "observation_id",
            sa.Integer,
            sa.ForeignKey("theme_development_observations.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "theme_id",
            sa.Integer,
            sa.ForeignKey("theme_clusters.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )
    op.create_index(
        "ix_theme_development_themes_theme_id", "theme_development_themes", ["theme_id"]
    )
    op.create_table(
        "theme_development_work",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "content_item_id",
            sa.Integer,
            sa.ForeignKey("content_items.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("pipeline", sa.String(20), nullable=False),
        sa.Column("revision", sa.String(64), nullable=False),
        sa.Column("source_marker", sa.Integer, nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False),
        sa.Column("error_code", sa.String(80)),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("claim_token", sa.String(64)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.UniqueConstraint(
            "content_item_id", "pipeline", "revision", name="uq_theme_development_work"
        ),
    )
    op.create_index(
        "ix_theme_development_work_status", "theme_development_work", ["status"]
    )


def downgrade():
    op.drop_column("theme_metrics", "grouping_version")
    for table in [
        "theme_development_work",
        "theme_development_themes",
        "theme_development_observations",
        "theme_development_events",
        "theme_equivalence_operations",
    ]:
        op.drop_table(table)
