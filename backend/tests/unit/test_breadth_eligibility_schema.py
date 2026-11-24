from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from app.models.market_breadth import MarketBreadth
from sqlalchemy import Float, Integer, String


REVISION_2_INTEGER_COLUMNS = {
    "advancing_count",
    "declining_count",
    "unchanged_count",
    "new_high_52week_count",
    "new_low_52week_count",
    "t2108_count",
    "atr_10x_extension_count",
    "broad_universe_count",
    "advance_decline_eligible_count",
    "stockbee_daily_eligible_count",
    "stockbee_month_eligible_count",
    "stockbee_34day_eligible_count",
    "stockbee_quarter_eligible_count",
    "t2108_eligible_count",
    "high_low_52week_eligible_count",
    "atr_extension_eligible_count",
    "calculation_revision",
}


def test_market_breadth_has_nullable_eligibility_signature():
    column = MarketBreadth.__table__.c.eligibility_signature

    assert column.type.length == 64
    assert column.nullable is True


def test_breadth_eligibility_migration_follows_current_head():
    path = (
        Path(__file__).resolve().parents[2]
        / "alembic/versions/20260808_0027_add_breadth_eligibility_signature.py"
    )
    spec = spec_from_file_location("breadth_eligibility_migration", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.revision == "20260808_0027"
    assert module.down_revision == "20260805_0026"


def test_market_breadth_has_nullable_revision_2_columns():
    columns = MarketBreadth.__table__.c

    for name in REVISION_2_INTEGER_COLUMNS:
        assert isinstance(columns[name].type, Integer)
        assert columns[name].nullable is True
    assert isinstance(columns.t2108_pct.type, Float)
    assert columns.t2108_pct.nullable is True
    assert isinstance(columns.stockbee_eligibility_signature.type, String)
    assert columns.stockbee_eligibility_signature.type.length == 64
    assert columns.stockbee_eligibility_signature.nullable is True


def test_shared_breadth_migration_follows_current_head():
    path = (
        Path(__file__).resolve().parents[2]
        / "alembic/versions/20260825_0031_add_shared_breadth_metrics.py"
    )
    spec = spec_from_file_location("shared_breadth_migration", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.revision == "20260825_0031"
    assert module.down_revision == "20260823_0030"


PARTICIPATION_COLUMNS = {
    "above_sma200_count": Integer,
    "above_sma200_pct": Float,
    "down_20pct_from_high_count": Integer,
    "down_20pct_from_high_pct": Float,
    "up_4pct_near_high_count": Integer,
    "sma200_eligible_count": Integer,
    "off_high_eligible_count": Integer,
}
SHORT_TERM_COLUMNS = {
    "above_sma20_count": Integer,
    "above_sma20_pct": Float,
    "above_sma50_count": Integer,
    "above_sma50_pct": Float,
    "trend_template_entry_count": Integer,
    "sma20_eligible_count": Integer,
    "sma50_eligible_count": Integer,
    "trend_template_entry_eligible_count": Integer,
}


def test_market_breadth_has_nullable_participation_columns():
    columns = MarketBreadth.__table__.c

    for name, column_type in {**PARTICIPATION_COLUMNS, **SHORT_TERM_COLUMNS}.items():
        assert isinstance(columns[name].type, column_type)
        assert columns[name].nullable is True


@pytest.mark.parametrize(
    ("filename", "revision", "down_revision", "columns"),
    [
        (
            "20261123_0077_breadth_participation_columns.py",
            "20261123_0077",
            "20261122_0076",
            PARTICIPATION_COLUMNS,
        ),
        (
            "20261124_0079_breadth_short_term_and_template_columns.py",
            "20261124_0079",
            "20261123_0078",
            SHORT_TERM_COLUMNS,
        ),
    ],
)
def test_participation_migrations_add_the_model_columns_after_their_head(
    filename, revision, down_revision, columns
):
    path = Path(__file__).resolve().parents[2] / "alembic/versions" / filename
    spec = spec_from_file_location(f"breadth_migration_{revision}", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.revision == revision
    assert module.down_revision == down_revision
    assert {name for name, _ in module._COLUMNS} == set(columns)
