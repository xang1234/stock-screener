"""Remove duplicated membership and mention-marker freshness state."""

import sqlalchemy as sa
from alembic import op

revision = "20260912_0044"
down_revision = "20260912_0043"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    observations = sa.table(
        "theme_development_observations",
        sa.column("id", sa.Integer),
        sa.column("theme_ids", sa.JSON),
        sa.column("facts", sa.JSON),
    )
    links = sa.table(
        "theme_development_themes",
        sa.column("observation_id", sa.Integer),
        sa.column("theme_id", sa.Integer),
    )
    # Existing association rows are authoritative. Import any missing legacy JSON
    # memberships before dropping the redundant representation.
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    insert = pg_insert if connection.dialect.name == "postgresql" else sqlite_insert
    for row in connection.execute(sa.select(observations)).yield_per(1000):
        for theme_id in set(row.theme_ids or []):
            connection.execute(
                insert(links)
                .values(observation_id=row.id, theme_id=theme_id)
                .on_conflict_do_nothing()
            )
        facts = dict(row.facts or {})
        facts.pop("theme_ids", None)
        connection.execute(
            observations.update().where(observations.c.id == row.id).values(facts=facts)
        )
    op.drop_column("theme_development_observations", "theme_ids")
    with op.batch_alter_table("theme_development_work") as batch:
        batch.add_column(
            sa.Column(
                "checked_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            )
        )
    op.drop_column("theme_development_work", "source_marker")


def downgrade():
    op.add_column(
        "theme_development_work",
        sa.Column("source_marker", sa.Integer, nullable=False, server_default="0"),
    )
    op.drop_column("theme_development_work", "checked_at")
    op.add_column("theme_development_observations", sa.Column("theme_ids", sa.JSON))
    connection = op.get_bind()
    observations = sa.table(
        "theme_development_observations",
        sa.column("id", sa.Integer),
        sa.column("theme_ids", sa.JSON),
        sa.column("facts", sa.JSON),
    )
    links = sa.table(
        "theme_development_themes",
        sa.column("observation_id", sa.Integer),
        sa.column("theme_id", sa.Integer),
    )
    for row in connection.execute(sa.select(observations)).yield_per(1000):
        ids = sorted(
            connection.execute(
                sa.select(links.c.theme_id).where(links.c.observation_id == row.id)
            ).scalars()
        )
        connection.execute(
            observations.update()
            .where(observations.c.id == row.id)
            .values(theme_ids=ids, facts={**row.facts, "theme_ids": ids})
        )
    with op.batch_alter_table("theme_development_observations") as batch:
        batch.alter_column("theme_ids", existing_type=sa.JSON(), nullable=False)
