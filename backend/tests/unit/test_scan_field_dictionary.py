from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from app.domain.scanning.models import ScanResultItemDomain
from app.schemas.scanning import ScanResultItem
from app.services.static_scan_bundle_exporter import StaticScanBundleExporter

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"

NOT_ON_PUBLISHED_ROWS = {
    "se_explain": "serialize_scan_row excludes this setup payload",
    "se_candidates": "serialize_scan_row excludes this setup payload",
    "se_atr14_pct_trend": "filter contract only, absent from ScanResultItem",
    "se_bb_width_pct": "filter contract only, absent from ScanResultItem",
    "se_rs": "filter contract only, absent from ScanResultItem",
    "se_rs_vs_spy_65d": "filter contract only, absent from ScanResultItem",
    "se_rs_vs_spy_trend_20d": "filter contract only, absent from ScanResultItem",
}


def _dictionary() -> dict:
    return json.loads((CONTRACTS / "scan_field_dictionary.json").read_text())["fields"]


def _exported_row_keys() -> set[str]:
    row = StaticScanBundleExporter.serialize_scan_row(
        ScanResultItemDomain(
            symbol="AAA",
            composite_score=50.0,
            rating="Pass",
            current_price=10.0,
            screener_outputs={},
            screeners_run=[],
            composite_method="weighted_average",
            screeners_passed=0,
            screeners_total=0,
        )
    )
    StaticScanBundleExporter.annotate_percentile_ranks([row])
    as_of = date(2026, 1, 2)
    publication = SimpleNamespace(
        snapshot=SimpleNamespace(formula_version="v", as_of_date=as_of),
        market_rs_run_id=1,
        universe_size=2,
    )
    exporter = StaticScanBundleExporter(
        rs_identity_resolver=lambda run, ranking_date: SimpleNamespace(publication=publication)
    )
    return {*row, *exporter._rs_metadata(SimpleNamespace(as_of_date=as_of))}


def _filter_contract_fields() -> set[str]:
    contract = json.loads((CONTRACTS / "scan_filter_fields.json").read_text())
    return {entry["field"] for entry in contract["fields"]}


def _scan_fields() -> set[str]:
    return {*ScanResultItem.model_fields, *_exported_row_keys(), *_filter_contract_fields()}


def test_exported_row_keys_include_what_only_the_exporter_adds():
    assert {
        "perf_week", "week_52_high_distance", "pct_day", "pct_week", "pct_month",
        "rs_formula_version", "market_rs_run_id", "rs_as_of_date", "rs_universe_size",
    } <= _exported_row_keys()


def test_every_scan_field_is_defined_or_excluded_with_a_reason():
    assert sorted(_scan_fields() - _dictionary().keys() - NOT_ON_PUBLISHED_ROWS.keys()) == []


def test_exclusions_stay_true():
    assert sorted(NOT_ON_PUBLISHED_ROWS.keys() & _dictionary().keys()) == []
    assert sorted(NOT_ON_PUBLISHED_ROWS.keys() - _scan_fields()) == []
    assert sorted(NOT_ON_PUBLISHED_ROWS.keys() & _exported_row_keys()) == []
