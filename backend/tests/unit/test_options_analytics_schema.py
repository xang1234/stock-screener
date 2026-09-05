from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import sqlalchemy as sa

from app.infra.db.models.options_analytics import (
    OptionsAnalyticsPointer,
    OptionsAnalyticsRun,
    OptionsAnalyticsRunItem,
    OptionsAnalyticsStrikePoint,
)


def _constraint_names(table) -> set[str]:
    return {constraint.name for constraint in table.constraints if constraint.name}


def test_options_models_define_run_item_strike_and_pointer_identity() -> None:
    assert OptionsAnalyticsRun.__table__.name == "options_analytics_runs"
    assert OptionsAnalyticsRunItem.__table__.name == "options_analytics_run_items"
    assert (
        OptionsAnalyticsStrikePoint.__table__.name == "options_analytics_strike_points"
    )
    assert OptionsAnalyticsPointer.__table__.name == "options_analytics_pointers"

    assert "uq_options_run_signature_attempt" in _constraint_names(
        OptionsAnalyticsRun.__table__
    )
    assert "uq_options_run_item_symbol" in _constraint_names(
        OptionsAnalyticsRunItem.__table__
    )
    assert "uq_options_strike_item_strike" in _constraint_names(
        OptionsAnalyticsStrikePoint.__table__
    )
    assert OptionsAnalyticsPointer.__table__.primary_key.columns.keys() == [
        "market",
        "calculation_version",
    ]


def test_options_models_keep_sortable_metrics_typed_and_diagnostics_json() -> None:
    columns = OptionsAnalyticsRunItem.__table__.columns

    for name in (
        "max_pain",
        "net_gex",
        "gamma_flip",
        "atm_iv",
        "vrp",
        "activity_intensity",
        "iv_percentile",
        "iv_rank",
        "max_pain_change_5",
        "activity_intensity_change_5",
        "call_put_volume_ratio",
        "volume_oi_ratio",
        "near_spot_volume_concentration",
        "near_spot_open_interest_concentration",
    ):
        assert isinstance(columns[name].type, sa.Float)
    for name in (
        "call_open_interest",
        "put_open_interest",
        "call_volume",
        "put_volume",
    ):
        assert isinstance(columns[name].type, sa.BigInteger)
    assert isinstance(columns["activity_rank"].type, sa.Integer)
    assert isinstance(columns["core_valid"].type, sa.Boolean)
    for name in ("evidence_json", "assumptions_json", "warnings_json", "reasons_json"):
        assert isinstance(columns[name].type, sa.JSON)


def test_options_foreign_keys_have_owned_cascades_but_pointer_does_not() -> None:
    item_fk = next(iter(OptionsAnalyticsRunItem.__table__.c.run_id.foreign_keys))
    strike_fk = next(iter(OptionsAnalyticsStrikePoint.__table__.c.item_id.foreign_keys))
    pointer_fk = next(iter(OptionsAnalyticsPointer.__table__.c.run_id.foreign_keys))

    assert item_fk.ondelete == "CASCADE"
    assert strike_fk.ondelete == "CASCADE"
    assert pointer_fk.ondelete != "CASCADE"


def test_options_migration_extends_the_verified_single_head() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "20260904_0034_add_options_analytics.py"
    )
    spec = spec_from_file_location("options_analytics_migration", path)
    assert spec is not None and spec.loader is not None
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)

    assert migration.revision == "20260904_0034"
    assert migration.down_revision == "20260829_0033"
