"""Mark Exchange Traded Funds and closed-end funds as not common stock.

Universe ingestion wrote is_common_stock = true for every row. On the live
database on 2026-11-03 that held for all 4,105 US Exchange Traded Fund rows and
335 Closed-End Fund rows, so they counted as common stock in RS percentiles and
breadth. Ingestion now classifies them on insert and refresh, and this backfills
rows already stored. The prefixes are those of
app.domain.universe.ingestion.FUND_INDUSTRY_PREFIXES at this revision.
"""

from alembic import op

revision = "20261103_0074"
down_revision = "20261101_0073"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE stock_universe
           SET is_common_stock = false
         WHERE is_common_stock
           AND (industry LIKE 'Exchange Traded Fund%' OR industry LIKE 'Closed-End Fund%')
        """
    )


def downgrade() -> None:
    # Marking funds as common stock again would restore the bug.
    pass
