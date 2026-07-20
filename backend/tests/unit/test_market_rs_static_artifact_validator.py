"""Strict validation coverage for staged canonical Market RS artifacts."""

from __future__ import annotations

from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.services.market_rs_static_artifact_validator import (
    MarketRsStaticArtifactValidator,
    StaticRsArtifactDocuments,
)
from app.services.static_groups_rrg_export import (
    StaticGroupsRRGUnavailableError,
    StaticGroupsRRGUnavailableReason,
)
from app.services.static_site_export_service import STATIC_SITE_SCHEMA_VERSION
from app.services.static_site_export_service import SCAN_BUNDLE_SCHEMA_VERSION


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _staged_bundle(
    tmp_path,
    *,
    market: str = "US",
    groups_available: bool = True,
    second_row_metadata: dict | None = None,
):
    market_lower = market.lower()
    market_dir = tmp_path / "markets" / market_lower
    identity = {
        "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": 42,
        "rs_as_of_date": "2026-04-10",
        "rs_universe_size": 2,
    }
    _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "markets": {market: dict(identity)},
        },
    )
    groups_payload = (
        {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            **identity,
            "payload": {"rankings": {"rankings": []}},
        }
        if groups_available
        else {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "available": False,
            "message": f"No group rankings are available for {market}.",
            "payload": {},
        }
    )
    _write_json(
        market_dir / "groups.json",
        groups_payload,
    )
    rows = [
        {
            "symbol": "AAA",
            "rs_rating": 90,
            "rs_rating_1m": 88,
            "rs_rating_3m": 89,
            "rs_rating_12m": 91,
            **identity,
        },
        {
            "symbol": "BBB",
            "rs_rating": 80,
            "rs_rating_1m": 78,
            "rs_rating_3m": 79,
            "rs_rating_12m": 81,
            **identity,
            **(second_row_metadata or {}),
        },
    ]
    chunks = []
    for index, row in enumerate(rows, start=1):
        relative = f"markets/{market_lower}/scan/chunks/chunk-{index:04d}.json"
        chunks.append({"path": relative, "count": 1})
        _write_json(
            tmp_path / relative,
            {
                "schema_version": SCAN_BUNDLE_SCHEMA_VERSION,
                "run_id": 99,
                "chunk_index": index,
                **identity,
                "rows": [row],
            },
        )
    _write_json(
        market_dir / "scan" / "manifest.json",
        {
            "schema_version": SCAN_BUNDLE_SCHEMA_VERSION,
            "run_id": 99,
            "rows_total": 2,
            "chunks": chunks,
            **identity,
        },
    )
    return market_dir


def _latest_run():
    return SimpleNamespace(
        id=42,
        eligible_symbol_count=2,
        rows=[
            SimpleNamespace(
                symbol="AAA", overall_rs=90, rs_1m=88, rs_3m=89, rs_12m=91
            ),
            SimpleNamespace(
                symbol="BBB", overall_rs=80, rs_1m=78, rs_3m=79, rs_12m=81
            ),
        ],
    )


def test_bundle_fingerprint_covers_market_files_not_only_root_manifest(tmp_path):
    market_dir = _staged_bundle(tmp_path)

    before = MarketRsStaticArtifactValidator.bundle_fingerprint(
        tmp_path,
        market="US",
    )
    groups_path = market_dir / "groups.json"
    groups_path.write_text(groups_path.read_text() + "\n", encoding="utf-8")
    after = MarketRsStaticArtifactValidator.bundle_fingerprint(
        tmp_path,
        market="US",
    )

    assert before.sha256 != after.sha256
    assert "markets/us/groups.json" in before.files


def test_validate_checks_every_scan_shard_and_requires_exact_row_identity(
    tmp_path,
    monkeypatch,
):
    _staged_bundle(
        tmp_path,
        second_row_metadata={"market_rs_run_id": None},
    )
    validator = MarketRsStaticArtifactValidator()
    monkeypatch.setattr(validator, "_validate_group_parity", lambda *a, **k: None)
    monkeypatch.setattr(validator, "_validate_rrg", lambda *a, **k: "available")

    result = validator.validate(
        MagicMock(),
        market="US",
        through_date=date(2026, 4, 10),
        latest_run=_latest_run(),
        feature_run_id=99,
        static_staging_dir=tmp_path,
    )

    assert any(
        "BBB.market_rs_run_id" in error and "expected 42" in error
        for error in result.errors
    )
    assert result.bundle_fingerprint is not None


def test_stock_parity_allows_audited_rows_excluded_from_canonical_ratings():
    latest_run = SimpleNamespace(
        id=42,
        eligible_symbol_count=1,
        rows=[
            SimpleNamespace(
                symbol="AAA",
                overall_rs=90,
                rs_1m=88,
                rs_3m=89,
                rs_12m=91,
            )
        ],
    )
    identity = {
        "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": 42,
        "rs_as_of_date": "2026-04-10",
        "rs_universe_size": 1,
    }
    documents = StaticRsArtifactDocuments(
        manifest={},
        groups={},
        scan_rows=(
            {
                "symbol": "AAA",
                "rs_rating": 90,
                "rs_rating_1m": 88,
                "rs_rating_3m": 89,
                "rs_rating_12m": 91,
                **identity,
            },
            {
                "symbol": "YOUNG",
                "rs_rating": None,
                "rs_rating_1m": None,
                "rs_rating_3m": None,
                "rs_rating_12m": None,
                **identity,
            },
        ),
    )
    errors: list[str] = []

    MarketRsStaticArtifactValidator._validate_stock_parity(
        latest_run=latest_run,
        through_date=date(2026, 4, 10),
        documents=documents,
        errors=errors,
    )

    assert errors == []


def test_rrg_validation_accepts_market_where_rrg_is_not_enabled(
    tmp_path,
    monkeypatch,
):
    class _UnavailableSource:
        def __init__(self, **_kwargs):
            pass

        def build(self, **_kwargs):
            raise StaticGroupsRRGUnavailableError(
                section="CA rrg",
                reason_code=StaticGroupsRRGUnavailableReason.NOT_ENABLED,
                reason="RRG is not enabled for market CA.",
            )

    monkeypatch.setattr(
        "app.services.market_rs_static_artifact_validator."
        "StaticGroupsRRGDatabasePayloadSource",
        _UnavailableSource,
    )
    errors: list[str] = []

    status = MarketRsStaticArtifactValidator()._validate_rrg(
        MagicMock(),
        market="CA",
        through_date=date(2026, 4, 10),
        market_dir=tmp_path,
        errors=errors,
    )

    assert status == "not_enabled"
    assert errors == []


def test_validate_accepts_unavailable_groups_for_group_less_market(
    tmp_path,
    monkeypatch,
):
    _staged_bundle(tmp_path, market="DE", groups_available=False)
    validator = MarketRsStaticArtifactValidator()
    validate_group_parity = MagicMock()
    monkeypatch.setattr(validator, "_validate_group_parity", validate_group_parity)
    monkeypatch.setattr(validator, "_validate_rrg", lambda *a, **k: "not_enabled")

    result = validator.validate(
        MagicMock(),
        market="DE",
        through_date=date(2026, 4, 10),
        latest_run=_latest_run(),
        feature_run_id=99,
        static_staging_dir=tmp_path,
    )

    assert result.errors == ()
    assert result.rrg_status == "not_enabled"
    validate_group_parity.assert_not_called()
