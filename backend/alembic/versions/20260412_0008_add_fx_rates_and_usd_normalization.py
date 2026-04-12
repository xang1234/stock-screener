"""Add fx_rates table and USD-normalised columns on stock_fundamentals.

T3 of the ASIA fundamentals epic: compute USD comparables while retaining
local-currency originals.

Adds
----
- ``fx_rates`` table: append-only log of daily FX rates per
  (from_currency, to_currency, as_of_date, source).
- ``stock_fundamentals.market_cap_usd`` (BigInteger, indexed): USD-
  normalised market cap for cross-market ranking.
- ``stock_fundamentals.adv_usd`` (BigInteger, indexed): USD-denominated
  average dollar volume = ``avg_volume * (market_cap / shares_outstanding)
  * fx_rate``.
- ``stock_fundamentals.fx_metadata`` (JSONB): per-row FX snapshot
  (rate, from_currency, to_currency, as_of_date, source) for replay.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260412_0008"
down_revision = "20260412_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fx_rates",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("from_currency", sa.String(length=8), nullable=False),
        sa.Column("to_currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("rate", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "from_currency", "to_currency", "as_of_date", "source",
            name="uq_fx_rates_currency_date_source",
        ),
    )
    op.create_index(
        "ix_fx_rates_lookup",
        "fx_rates",
        ["from_currency", "to_currency", "as_of_date"],
        unique=False,
    )

    op.add_column(
        "stock_fundamentals",
        sa.Column("market_cap_usd", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("adv_usd", sa.BigInteger(), nullable=True),
    )
    # JSONB so T4 can filter on {"source": "..."} or {"rate": ...} without
    # deserialising every row.
    op.add_column(
        "stock_fundamentals",
        sa.Column("fx_metadata", postgresql.JSONB(), nullable=True),
    )
    # Indexes on the USD columns so T4 can rank/filter cross-market
    # without full scans.
    op.create_index(
        "ix_stock_fundamentals_market_cap_usd",
        "stock_fundamentals",
        ["market_cap_usd"],
        unique=False,
    )
    op.create_index(
        "ix_stock_fundamentals_adv_usd",
        "stock_fundamentals",
        ["adv_usd"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_stock_fundamentals_adv_usd", table_name="stock_fundamentals")
    op.drop_index("ix_stock_fundamentals_market_cap_usd", table_name="stock_fundamentals")
    op.drop_column("stock_fundamentals", "fx_metadata")
    op.drop_column("stock_fundamentals", "adv_usd")
    op.drop_column("stock_fundamentals", "market_cap_usd")
    op.drop_index("ix_fx_rates_lookup", table_name="fx_rates")
    op.drop_table("fx_rates")
