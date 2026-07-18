"""Add canonical, versioned Market relative-strength snapshots."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260718_0025"
down_revision = "20260701_0024"
branch_labels = None
depends_on = None


LEGACY_FORMULA_VERSION = "legacy-linear-v1"
CATALOG_MARKETS = (
    "US",
    "HK",
    "IN",
    "JP",
    "KR",
    "TW",
    "CN",
    "CA",
    "DE",
    "SG",
    "AU",
    "MY",
)
SQLITE_BATCH_NAMING_CONVENTION = {
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
}


def _unique_constraint_names_for_columns(
    bind,
    table_name: str,
    columns: tuple[str, ...],
    *,
    unnamed_sqlite_name: str,
) -> list[str]:
    names: list[str] = []
    for constraint in sa.inspect(bind).get_unique_constraints(table_name):
        if tuple(constraint.get("column_names") or ()) != columns:
            continue
        names.append(constraint.get("name") or unnamed_sqlite_name)
    return names


def _seed_legacy_formula_pointers(bind) -> None:
    inspector = sa.inspect(bind)
    markets = set(CATALOG_MARKETS)
    for table_name in ("stock_universe", "ibd_group_ranks"):
        if not inspector.has_table(table_name):
            continue
        rows = bind.execute(
            sa.text(
                f"SELECT DISTINCT market FROM {table_name} "
                "WHERE market IS NOT NULL"
            )
        )
        markets.update(str(row[0]).upper() for row in rows if row[0])

    pointer_table = sa.table(
        "market_rs_formula_pointers",
        sa.column("market", sa.String(length=8)),
        sa.column("formula_version", sa.String(length=64)),
    )
    op.bulk_insert(
        pointer_table,
        [
            {"market": market, "formula_version": LEGACY_FORMULA_VERSION}
            for market in sorted(markets)
        ],
    )


def upgrade() -> None:
    op.create_table(
        "market_rs_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("market", sa.String(length=8), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("formula_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("benchmark_symbol", sa.String(length=32), nullable=False),
        sa.Column("benchmark_as_of_date", sa.Date(), nullable=False),
        sa.Column("universe_hash", sa.String(length=64), nullable=False),
        sa.Column("expected_symbol_count", sa.Integer(), nullable=False),
        sa.Column("eligible_symbol_count", sa.Integer(), nullable=False),
        sa.Column("excluded_symbol_count", sa.Integer(), nullable=False),
        sa.Column("diagnostics_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_market_rs_run_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "market", "as_of_date", "formula_version", name="uq_market_rs_run"
        ),
    )
    op.create_index(
        "ix_market_rs_run_lookup",
        "market_rs_runs",
        ["market", "formula_version", "as_of_date", "status"],
        unique=False,
    )

    op.create_table(
        "stock_rs_snapshots",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("overall_rs", sa.SmallInteger(), nullable=False),
        sa.Column("rs_1m", sa.SmallInteger(), nullable=False),
        sa.Column("rs_3m", sa.SmallInteger(), nullable=False),
        sa.Column("rs_6m", sa.SmallInteger(), nullable=False),
        sa.Column("rs_9m", sa.SmallInteger(), nullable=False),
        sa.Column("rs_12m", sa.SmallInteger(), nullable=False),
        sa.Column("weighted_composite", sa.Float(), nullable=False),
        sa.Column("excess_return_1m", sa.Float(), nullable=False),
        sa.Column("excess_return_3m", sa.Float(), nullable=False),
        sa.Column("excess_return_6m", sa.Float(), nullable=False),
        sa.Column("excess_return_9m", sa.Float(), nullable=False),
        sa.Column("excess_return_12m", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "overall_rs BETWEEN 1 AND 99 AND rs_1m BETWEEN 1 AND 99 "
            "AND rs_3m BETWEEN 1 AND 99 AND rs_6m BETWEEN 1 AND 99 "
            "AND rs_9m BETWEEN 1 AND 99 AND rs_12m BETWEEN 1 AND 99",
            name="ck_stock_rs_rating_range",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["market_rs_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "symbol"),
    )
    op.create_index(
        "ix_stock_rs_symbol_run",
        "stock_rs_snapshots",
        ["symbol", "run_id"],
        unique=False,
    )

    op.create_table(
        "market_rs_formula_pointers",
        sa.Column("market", sa.String(length=8), nullable=False),
        sa.Column("formula_version", sa.String(length=64), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("market"),
    )

    with op.batch_alter_table("ibd_group_ranks") as batch_op:
        batch_op.add_column(sa.Column("avg_rs_rating_1m", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("avg_rs_rating_3m", sa.Float(), nullable=True))
        batch_op.add_column(
            sa.Column("rs_formula_version", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(sa.Column("market_rs_run_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_ibd_group_rank_market_rs_run",
            "market_rs_runs",
            ["market_rs_run_id"],
            ["id"],
            ondelete="SET NULL",
        )

    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE ibd_group_ranks SET rs_formula_version = :formula_version "
            "WHERE rs_formula_version IS NULL"
        ),
        {"formula_version": LEGACY_FORMULA_VERSION},
    )

    if bind.dialect.name == "sqlite":
        old_unique_names = _unique_constraint_names_for_columns(
            bind,
            "ibd_group_ranks",
            ("industry_group", "date", "market"),
            unnamed_sqlite_name="uq_ibd_group_ranks_industry_group_date_market",
        )
        with op.batch_alter_table(
            "ibd_group_ranks",
            naming_convention=SQLITE_BATCH_NAMING_CONVENTION,
        ) as batch_op:
            batch_op.alter_column(
                "rs_formula_version",
                existing_type=sa.String(length=64),
                nullable=False,
            )
            for name in old_unique_names:
                batch_op.drop_constraint(name, type_="unique")
            batch_op.create_unique_constraint(
                "uix_ibd_group_rank_market_date_formula",
                ["industry_group", "date", "market", "rs_formula_version"],
            )
    else:
        op.alter_column(
            "ibd_group_ranks",
            "rs_formula_version",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        op.drop_constraint(
            "uix_ibd_group_rank_market_date", "ibd_group_ranks", type_="unique"
        )
        op.create_unique_constraint(
            "uix_ibd_group_rank_market_date_formula",
            "ibd_group_ranks",
            ["industry_group", "date", "market", "rs_formula_version"],
        )

    _seed_legacy_formula_pointers(bind)


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "DELETE FROM ibd_group_ranks "
            "WHERE rs_formula_version <> :formula_version"
        ),
        {"formula_version": LEGACY_FORMULA_VERSION},
    )

    if bind.dialect.name == "sqlite":
        with op.batch_alter_table(
            "ibd_group_ranks",
            naming_convention=SQLITE_BATCH_NAMING_CONVENTION,
        ) as batch_op:
            batch_op.drop_constraint(
                "uix_ibd_group_rank_market_date_formula", type_="unique"
            )
            batch_op.create_unique_constraint(
                "uix_ibd_group_rank_market_date",
                ["industry_group", "date", "market"],
            )
            batch_op.drop_constraint(
                "fk_ibd_group_rank_market_rs_run", type_="foreignkey"
            )
            batch_op.drop_column("market_rs_run_id")
            batch_op.drop_column("rs_formula_version")
            batch_op.drop_column("avg_rs_rating_3m")
            batch_op.drop_column("avg_rs_rating_1m")
    else:
        op.drop_constraint(
            "uix_ibd_group_rank_market_date_formula",
            "ibd_group_ranks",
            type_="unique",
        )
        op.create_unique_constraint(
            "uix_ibd_group_rank_market_date",
            "ibd_group_ranks",
            ["industry_group", "date", "market"],
        )
        op.drop_constraint(
            "fk_ibd_group_rank_market_rs_run",
            "ibd_group_ranks",
            type_="foreignkey",
        )
        op.drop_column("ibd_group_ranks", "market_rs_run_id")
        op.drop_column("ibd_group_ranks", "rs_formula_version")
        op.drop_column("ibd_group_ranks", "avg_rs_rating_3m")
        op.drop_column("ibd_group_ranks", "avg_rs_rating_1m")

    op.drop_table("market_rs_formula_pointers")
    op.drop_index("ix_stock_rs_symbol_run", table_name="stock_rs_snapshots")
    op.drop_table("stock_rs_snapshots")
    op.drop_index("ix_market_rs_run_lookup", table_name="market_rs_runs")
    op.drop_table("market_rs_runs")
