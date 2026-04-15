"""Per-market telemetry alerts table (bead asia.10.2).

Stateful alert log with a partial unique index that enforces the
"at most one active alert per (market, metric_key)" hysteresis rule.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260415_0013"
down_revision = "20260415_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    json_type = postgresql.JSONB if dialect == "postgresql" else sa.JSON

    op.create_table(
        "market_telemetry_alerts",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("market", sa.String(length=8), nullable=False),
        sa.Column("metric_key", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=10), nullable=False),
        sa.Column("state", sa.String(length=12), nullable=False),
        sa.Column("owner", sa.String(length=64)),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("metrics", json_type()),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True)),
        sa.Column("acknowledged_by", sa.String(length=64)),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_telemetry_alerts_state_opened", "market_telemetry_alerts", ["state", "opened_at"])
    op.create_index("ix_telemetry_alerts_market_state", "market_telemetry_alerts", ["market", "state", "opened_at"])

    # Hysteresis enforcement: at most one active (open|acknowledged) alert per
    # (market, metric_key). Re-firing the same alert while it's still active is
    # a no-op; we update severity in place instead. The partial WHERE clause
    # keeps the constraint quiet for closed rows.
    if dialect == "postgresql":
        op.execute(
            """
            CREATE UNIQUE INDEX ux_telemetry_alerts_active
            ON market_telemetry_alerts (market, metric_key)
            WHERE state IN ('open', 'acknowledged')
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "postgresql":
        op.execute("DROP INDEX IF EXISTS ux_telemetry_alerts_active")
    op.drop_index("ix_telemetry_alerts_market_state", table_name="market_telemetry_alerts")
    op.drop_index("ix_telemetry_alerts_state_opened", table_name="market_telemetry_alerts")
    op.drop_table("market_telemetry_alerts")
