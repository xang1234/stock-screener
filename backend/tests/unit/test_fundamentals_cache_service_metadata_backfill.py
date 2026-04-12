from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from app.services.fundamentals_cache_service import FundamentalsCacheService
from app.services.fx_service import FXQuote


def test_backfill_repairs_all_derived_metadata_fields():
    fx_service = MagicMock()
    fx_service.get_usd_rate.return_value = FXQuote(
        from_currency="HKD",
        to_currency="USD",
        rate=0.128,
        as_of_date=date(2026, 4, 12),
        source="yfinance",
    )
    svc = FundamentalsCacheService(
        redis_client=None,
        session_factory=lambda: MagicMock(),
        fx_service=fx_service,
    )
    payload = {
        "market_cap": 1_000_000_000,
        "shares_outstanding": 1_000_000,
        "avg_volume": 2_000_000,
        "eps_rating": 75,
        "ipo_date": "2020-01-01",
        "first_trade_date": "2020-01-01",
        "sector": "Technology",
        "industry": "Software",
        "eps_growth_qq": 12.0,
        "sales_growth_qq": 10.0,
        "eps_growth_yy": 20.0,
        "sales_growth_yy": 18.0,
    }

    changed = svc._ensure_field_availability_metadata("0700.HK", payload, market="HK")

    assert changed is True
    assert "field_completeness_score" in payload
    assert isinstance(payload["field_provenance"], dict)
    assert isinstance(payload["field_availability"], dict)
    assert payload["market_cap_usd"] == 128_000_000
    assert payload["adv_usd"] == 256_000_000
    assert payload["fx_metadata"]["rate"] == 0.128


def test_backfill_repairs_none_placeholder_values():
    fx_service = MagicMock()
    fx_service.get_usd_rate.return_value = FXQuote(
        from_currency="HKD",
        to_currency="USD",
        rate=0.128,
        as_of_date=date(2026, 4, 12),
        source="yfinance",
    )
    svc = FundamentalsCacheService(
        redis_client=None,
        session_factory=lambda: MagicMock(),
        fx_service=fx_service,
    )
    payload = {
        "market_cap": 1_000_000_000,
        "shares_outstanding": 1_000_000,
        "avg_volume": 2_000_000,
        "eps_rating": 75,
        "ipo_date": "2020-01-01",
        "first_trade_date": "2020-01-01",
        "sector": "Technology",
        "industry": "Software",
        "eps_growth_qq": 12.0,
        "sales_growth_qq": 10.0,
        "eps_growth_yy": 20.0,
        "sales_growth_yy": 18.0,
        "field_completeness_score": None,
        "field_provenance": None,
        "field_availability": None,
        "market_cap_usd": None,
        "adv_usd": None,
        "fx_metadata": None,
    }

    changed = svc._ensure_field_availability_metadata("0700.HK", payload, market="HK")

    assert changed is True
    assert payload["field_completeness_score"] is not None
    assert isinstance(payload["field_provenance"], dict)
    assert isinstance(payload["field_availability"], dict)
    assert payload["market_cap_usd"] == 128_000_000
    assert payload["adv_usd"] == 256_000_000
    assert payload["fx_metadata"]["source"] == "yfinance"
