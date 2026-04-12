"""Integration tests: T3 FX normalisation end-to-end in the cache service.

Verifies that the per-write enrichment hook computes the USD columns and
attaches the reproducible fx_metadata snapshot.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from app.models.stock import StockFundamental
from app.services.fundamentals_cache_service import FundamentalsCacheService
from app.services.fx_service import FXQuote, FXService


def _make_fx(rate_by_currency):
    """FX service with a deterministic in-memory fetcher."""
    return FXService(
        rate_fetcher=lambda c: rate_by_currency.get(c),
        session_factory=lambda: _dummy_session(),
        redis_client=None,
    )


def _dummy_session():
    fake = MagicMock()
    fake.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
    fake.query.return_value.filter.return_value.first.return_value = None
    return fake


@pytest.fixture
def captured_record():
    captured = {"added_record": None}
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None

    def _capture_add(record):
        if isinstance(record, StockFundamental):
            captured["added_record"] = record

    fake_db.add.side_effect = _capture_add
    return fake_db, captured


class TestEnrichmentComputesUSDColumns:
    def test_hk_payload_normalises_market_cap_and_adv(self, captured_record):
        fake_db, captured = captured_record
        fx = _make_fx({"HKD": 0.128})
        service = FundamentalsCacheService(
            redis_client=None,
            session_factory=lambda: fake_db,
            fx_service=fx,
        )

        payload = {
            "market_cap": 1_000_000_000,  # HKD
            "shares_outstanding": 10_000_000,
            "avg_volume": 500_000,
        }
        # The public write path computes FX; drive it end-to-end.
        service._enrich_with_fx_normalization(payload, market="HK")
        service._store_in_database("0700.HK", payload, data_source="yfinance", market="HK")

        rec = captured["added_record"]
        assert rec is not None
        # market_cap_usd = 1_000_000_000 * 0.128 = 128_000_000
        assert rec.market_cap_usd == 128_000_000
        # price_local = 1e9 / 1e7 = 100 HKD; adv_usd = 500_000 * 100 * 0.128 = 6,400,000
        assert rec.adv_usd == 6_400_000
        # Snapshot is self-contained for replay.
        assert rec.fx_metadata == {
            "from_currency": "HKD",
            "to_currency": "USD",
            "rate": 0.128,
            "as_of_date": date.today().isoformat(),
            "source": "yfinance",
        }

    def test_us_payload_is_identity(self, captured_record):
        fake_db, captured = captured_record
        fx = _make_fx({})  # no fetches needed for USD
        service = FundamentalsCacheService(
            redis_client=None,
            session_factory=lambda: fake_db,
            fx_service=fx,
        )
        payload = {
            "market_cap": 5_000_000_000,
            "shares_outstanding": 100_000_000,
            "avg_volume": 2_000_000,
        }
        service._enrich_with_fx_normalization(payload, market="US")
        service._store_in_database("AAPL", payload, data_source="yfinance", market="US")

        rec = captured["added_record"]
        assert rec.market_cap_usd == 5_000_000_000
        # price = 50 USD; adv_usd = 2M * 50 = 100M
        assert rec.adv_usd == 100_000_000
        assert rec.fx_metadata["rate"] == 1.0
        assert rec.fx_metadata["source"] == "identity"
        assert rec.fx_metadata["from_currency"] == "USD"

    def test_missing_rate_leaves_usd_columns_null(self, captured_record):
        fake_db, captured = captured_record
        fx = _make_fx({})  # HKD returns None
        service = FundamentalsCacheService(
            redis_client=None,
            session_factory=lambda: fake_db,
            fx_service=fx,
        )
        payload = {
            "market_cap": 1_000_000_000,
            "shares_outstanding": 10_000_000,
            "avg_volume": 500_000,
        }
        service._enrich_with_fx_normalization(payload, market="HK")
        service._store_in_database("0700.HK", payload, data_source="yfinance", market="HK")

        rec = captured["added_record"]
        assert rec.market_cap_usd is None
        assert rec.adv_usd is None
        # fx_metadata still records the attempted-but-failed conversion so
        # operators can audit why the column is NULL.
        assert rec.fx_metadata["source"] == "unavailable"
        assert rec.fx_metadata["rate"] is None
        assert rec.fx_metadata["from_currency"] == "HKD"

    def test_missing_shares_outstanding_skips_adv_only(self, captured_record):
        fake_db, captured = captured_record
        fx = _make_fx({"HKD": 0.128})
        service = FundamentalsCacheService(
            redis_client=None,
            session_factory=lambda: fake_db,
            fx_service=fx,
        )
        payload = {
            "market_cap": 1_000_000_000,
            "shares_outstanding": None,
            "avg_volume": 500_000,
        }
        service._enrich_with_fx_normalization(payload, market="HK")
        service._store_in_database("0700.HK", payload, data_source="yfinance", market="HK")

        rec = captured["added_record"]
        # market_cap_usd still computed (doesn't need shares_outstanding).
        assert rec.market_cap_usd == 128_000_000
        # adv_usd requires shares_outstanding to derive price — NULL here.
        assert rec.adv_usd is None
        assert rec.fx_metadata["rate"] == 0.128
