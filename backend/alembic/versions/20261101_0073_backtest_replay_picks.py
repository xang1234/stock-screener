"""Store point-in-time replayed scan picks for the Backtest page.

Additive. The table starts empty and fills when an operator runs
app.scripts.replay_scan_picks; until then the replay source reports
no_matching_replay_picks.
"""

import sqlalchemy as sa

from alembic import op

revision = "20261101_0073"
down_revision = "20261028_0072"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "backtest_replay_picks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("market", sa.String(length=8), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("rank", sa.SmallInteger(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("minervini_score", sa.Float(), nullable=False),
        sa.Column("rs_rating", sa.SmallInteger(), nullable=False),
        sa.Column("stage", sa.SmallInteger(), nullable=True),
        sa.Column("definition_version", sa.String(length=64), nullable=False),
        sa.Column("market_rs_run_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "market", "as_of_date", "definition_version", "rank",
            name="uq_backtest_replay_pick_rank",
        ),
    )
    op.create_index(
        "ix_backtest_replay_picks_window",
        "backtest_replay_picks",
        ["market", "definition_version", "as_of_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_backtest_replay_picks_window", table_name="backtest_replay_picks")
    op.drop_table("backtest_replay_picks")
