"""Per-market telemetry event log + daily aggregating view (bead asia.10.1).

Schema-versioned event log for per-market freshness, drift, benchmark age,
extraction success, and completeness distribution. Append-only writes from
inline emission hooks; reads via the daily-aggregated view.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260415_0012"
down_revision = "20260411_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    json_type = postgresql.JSONB if dialect == "postgresql" else sa.JSON

    op.create_table(
        "market_telemetry_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("market", sa.String(length=8), nullable=False),
        sa.Column("metric_key", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.SmallInteger, nullable=False),
        sa.Column("payload", json_type(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Index("ix_market_telemetry_market_recorded", "market", "recorded_at"),
        sa.Index("ix_market_telemetry_metric_recorded", "metric_key", "recorded_at"),
        # Dedicated index for the cleanup DELETE so it doesn't seq-scan the whole
        # table when the 15d retention sweep runs from weekly_full_refresh.
        sa.Index("ix_market_telemetry_recorded_at", "recorded_at"),
    )

    # Daily aggregating view. Postgres-only; SQLite tests use the table directly.
    # ``schema_version`` is aggregated (MAX), not grouped by — otherwise a
    # mid-day schema bump would produce two rows for the same (market, metric, day)
    # and silently double-count event counts in dashboards.
    if dialect == "postgresql":
        op.execute(
            """
            CREATE OR REPLACE VIEW v_market_telemetry_daily AS
            SELECT
                market,
                metric_key,
                date_trunc('day', recorded_at)::date AS day,
                MAX(schema_version) AS schema_version,
                COUNT(*) AS event_count,
                MIN(recorded_at) AS first_at,
                MAX(recorded_at) AS last_at,
                (ARRAY_AGG(payload ORDER BY recorded_at DESC))[1] AS last_payload
            FROM market_telemetry_events
            GROUP BY market, metric_key, date_trunc('day', recorded_at)
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "postgresql":
        op.execute("DROP VIEW IF EXISTS v_market_telemetry_daily")
    op.drop_index("ix_market_telemetry_recorded_at", table_name="market_telemetry_events")
    op.drop_index("ix_market_telemetry_metric_recorded", table_name="market_telemetry_events")
    op.drop_index("ix_market_telemetry_market_recorded", table_name="market_telemetry_events")
    op.drop_table("market_telemetry_events")
