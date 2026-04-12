"""Integration tests proving T2 metadata is persisted by the cache storage path.

These tests are the end-to-end check that ``StockScreenClaude-asia.5.2``'s
acceptance criterion — *completeness score computed and queryable* — is
actually wired up: when something writes via ``store_all_caches`` /
``store``, the two new columns are populated with market-aware values.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.models.stock import StockFundamental
from app.services.fundamentals_cache_service import FundamentalsCacheService
from app.services.field_capability_registry import REASON_CODE_NON_US_GAP
from app.services.fundamentals_completeness import (
    CORE_FIELDS,
    STANDARD_FIELDS,
)


def _full_yfinance_payload() -> dict:
    return {f: 1.0 for f in CORE_FIELDS | STANDARD_FIELDS}


@pytest.fixture
def captured_record():
    """Return a (service, captured) pair. ``captured`` dict gets populated
    by the fake DB with the record that was added/updated."""
    captured = {"added_record": None}

    fake_db = MagicMock()
    # No existing record — forces the insert path.
    fake_db.query.return_value.filter.return_value.first.return_value = None

    def _capture_add(record):
        # Only capture the StockFundamental row; ownership updates also
        # call db.add() but we don't care about those here.
        if isinstance(record, StockFundamental):
            captured["added_record"] = record

    fake_db.add.side_effect = _capture_add

    service = FundamentalsCacheService(
        redis_client=None,
        session_factory=lambda: fake_db,
    )
    return service, captured


class TestStorageWritesCompletenessAndProvenance:
    def test_hk_full_yfinance_payload_scores_100_and_tags_yfinance(
        self, captured_record
    ):
        service, captured = captured_record
        payload = _full_yfinance_payload()

        service._store_in_database(
            "0700.HK", payload, data_source="yfinance", market="HK"
        )

        record = captured["added_record"]
        assert record is not None
        assert record.field_completeness_score == 100
        # Provenance dict should exist with at least one yfinance-tagged field.
        assert isinstance(record.field_provenance, dict)
        assert record.field_provenance.get("market_cap") == "yfinance"
        # Enhanced (finviz-only) fields must not appear for HK.
        assert "short_float" not in record.field_provenance

    def test_us_yfinance_only_payload_scores_below_100(self, captured_record):
        """US stocks are expected to have finviz fields too — missing finviz
        data should lower the score even when yfinance data is complete."""
        service, captured = captured_record
        payload = _full_yfinance_payload()

        service._store_in_database(
            "AAPL", payload, data_source="yfinance", market="US"
        )
        record = captured["added_record"]
        assert 0 < record.field_completeness_score < 100

    def test_empty_payload_scores_zero(self, captured_record):
        service, captured = captured_record

        service._store_in_database(
            "XXX", {}, data_source="yfinance", market="US"
        )
        record = captured["added_record"]
        assert record.field_completeness_score == 0
        assert record.field_provenance == {}

    def test_market_is_resolved_when_not_passed(self, captured_record, monkeypatch):
        """When caller omits ``market``, the service falls back to
        ``_resolve_market`` — verify the hook is honoured."""
        service, captured = captured_record
        # ``raising=True`` (default) errors if _resolve_market is renamed,
        # catching refactor breakage that plain assignment would miss.
        monkeypatch.setattr(service, "_resolve_market", lambda symbol: "HK")

        service._store_in_database("0700.HK", _full_yfinance_payload(), data_source="yfinance")
        record = captured["added_record"]
        # HK with full yfinance payload = 100% completeness.
        assert record.field_completeness_score == 100

    def test_hk_missing_ownership_fields_emit_t6_reason_codes(self, captured_record):
        service, _ = captured_record
        payload = _full_yfinance_payload()
        payload.pop("institutional_ownership", None)
        payload.pop("insider_ownership", None)

        service._enrich_with_quality_metadata("0700.HK", payload, market="HK")

        availability = payload.get("field_availability")
        assert isinstance(availability, dict)
        assert availability["institutional_ownership"]["reason_code"] == REASON_CODE_NON_US_GAP
        assert availability["insider_ownership"]["reason_code"] == REASON_CODE_NON_US_GAP
