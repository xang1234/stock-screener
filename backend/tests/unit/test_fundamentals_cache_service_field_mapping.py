"""Pin the StockFundamental <-> payload field mapping across insert, update and both reads."""
from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.stock import StockFundamental
from app.services.fundamentals_cache_service import FundamentalsCacheService

HISTORY = [
    {"report_ts": "2026-07-22T16:00:00-04:00", "eps_estimate": 1.73, "reported_eps": 1.54, "surprise_pct": -10.88},
]

FULL_PAYLOAD = {
    "market_cap": 1_000_000_001,
    "shares_outstanding": 2_000_000_002,
    "avg_volume": 3_000_003,
    "relative_volume": 1.04,
    "pe_ratio": 2.05,
    "forward_pe": 2.06,
    "peg_ratio": 2.07,
    "price_to_book": 2.08,
    "price_to_sales": 2.09,
    "price_to_cash": 2.10,
    "price_to_fcf": 2.11,
    "ev_ebitda": 2.12,
    "ev_sales": 2.13,
    "target_price": 2.14,
    "eps_current": 3.01,
    "eps_next_y": 3.02,
    "eps_next_5y": 3.03,
    "eps_next_q": 3.04,
    "eps_growth_qq": 3.05,
    "eps_growth_quarterly": 3.06,
    "eps_growth_yy": 3.07,
    "eps_growth_annual": 3.08,
    "revenue_current": 4_000_000_009,
    "revenue_growth": 3.10,
    "sales_past_5y": 3.11,
    "sales_growth_yy": 3.12,
    "sales_growth_qq": 3.13,
    "recent_quarter_date": "2026-Q2",
    "previous_quarter_date": "2026-Q1",
    "growth_reporting_cadence": "quarterly",
    "growth_metric_basis": "quarterly_qoq",
    "growth_comparable_period_date": "2025-06-30",
    "growth_reference_gap_days": 91,
    "next_earnings_date": "2026-11-05",
    "event_calendar_as_of_date": "2026-10-19",
    "recent_earnings": HISTORY,
    "profit_margin": 4.01,
    "operating_margin": 4.02,
    "gross_margin": 4.03,
    "roe": 4.04,
    "roa": 4.05,
    "roic": 4.06,
    "current_ratio": 5.01,
    "quick_ratio": 5.02,
    "debt_to_equity": 5.03,
    "lt_debt_to_equity": 5.04,
    "insider_ownership": 6.01,
    "insider_transactions": 6.02,
    "institutional_ownership": 6.03,
    "institutional_transactions": 6.04,
    "short_float": 6.06,
    "short_ratio": 6.07,
    "short_interest": 6_000_008,
    "beta": 7.01,
    "rsi_14": 7.02,
    "atr_14": 7.03,
    "sma_20": 7.04,
    "sma_50": 7.05,
    "sma_200": 7.06,
    "volatility_week": 7.07,
    "volatility_month": 7.08,
    "perf_week": 8.01,
    "perf_month": 8.02,
    "perf_quarter": 8.03,
    "perf_half_year": 8.04,
    "perf_year": 8.05,
    "perf_ytd": 8.06,
    "dividend_ttm": 9.01,
    "dividend_yield": 9.02,
    "payout_ratio": 9.03,
    "week_52_high": 10.01,
    "week_52_high_distance": 10.02,
    "week_52_low": 10.03,
    "week_52_low_distance": 10.04,
    "sector": "Technology",
    "industry": "Semiconductors",
    "country": "USA",
    "employees": 12_345,
    "ipo_date": "2010-06-29",
    "first_trade_date_ms": 981_201_600_000,
    "description_yfinance": "yfinance description",
    "description_finviz": "finviz description",
    "recommendation": 1.8,
    "eps_5yr_cagr": 11.01,
    "eps_q1_yoy": 11.02,
    "eps_q2_yoy": 11.03,
    "eps_raw_score": 11.04,
    "eps_rating": 88,
    "eps_years_available": 5,
    "data_source": "payload-source",
    "finviz_snapshot_revision": "rev-1",
    "finviz_snapshot_at": "2026-10-18T20:00:00",
    "yahoo_profile_refreshed_at": "2026-10-17T21:00:00",
    "yahoo_statements_refreshed_at": "2026-10-16T22:00:00",
    "technicals_refreshed_at": "2026-10-15T23:00:00",
    "field_completeness_score": 77,
    "field_provenance": {"market_cap": "finviz"},
    "market_cap_usd": 1_000_000_012,
    "adv_usd": 13_000_013,
    "fx_metadata": {"from_currency": "USD", "to_currency": "USD", "rate": 1.0},
}

PARTIAL_UPDATE = {
    "market_cap": None,
    "avg_volume": 0,
    "relative_volume": None,
    "pe_ratio": 20.5,
    "eps_growth_qq": 30.5,
    "eps_growth_quarterly": 99.0,
    "eps_growth_annual": 98.0,
    "sector": "Healthcare",
    "first_trade_date_ms": 981_201_600_000,
    "finviz_snapshot_at": "not-a-timestamp",
    "technicals_refreshed_at": datetime(2026, 10, 20, 1, 2, 3),
    "market_cap_usd": None,
    "fx_metadata": {"rate": 0.5},
    "recommendation": 2.5,
    "field_completeness_score": 41,
    "field_provenance": {"pe_ratio": "yfinance"},
}

CALENDAR_UPDATE = {
    "event_calendar_as_of_date": "2026-10-20",
    "recent_earnings": [],
    "market_cap": 5,
    "field_completeness_score": 5,
    "field_provenance": {},
}

COLUMNS_AFTER_INSERT = {
    "symbol": "AAPL",
    "market_cap": 1_000_000_001,
    "shares_outstanding": 2_000_000_002,
    "avg_volume": 3_000_003,
    "relative_volume": 1.04,
    "pe_ratio": 2.05,
    "forward_pe": 2.06,
    "peg_ratio": 2.07,
    "price_to_book": 2.08,
    "price_to_sales": 2.09,
    "price_to_cash": 2.10,
    "price_to_fcf": 2.11,
    "ev_ebitda": 2.12,
    "ev_sales": 2.13,
    "target_price": 2.14,
    "eps_current": 3.01,
    "eps_next_y": 3.02,
    "eps_next_5y": 3.03,
    "eps_next_q": 3.04,
    "eps_growth_quarterly": 3.05,
    "eps_growth_annual": 3.07,
    "eps_growth_yy": 3.07,
    "revenue_current": 4_000_000_009,
    "revenue_growth": 3.10,
    "sales_past_5y": 3.11,
    "sales_growth_yy": 3.12,
    "sales_growth_qq": 3.13,
    "recent_quarter_date": "2026-Q2",
    "previous_quarter_date": "2026-Q1",
    "growth_reporting_cadence": "quarterly",
    "growth_metric_basis": "quarterly_qoq",
    "growth_comparable_period_date": "2025-06-30",
    "growth_reference_gap_days": 91,
    "next_earnings_date": date(2026, 11, 5),
    "event_calendar_as_of_date": date(2026, 10, 19),
    "recent_earnings": HISTORY,
    "profit_margin": 4.01,
    "operating_margin": 4.02,
    "gross_margin": 4.03,
    "roe": 4.04,
    "roa": 4.05,
    "roic": 4.06,
    "current_ratio": 5.01,
    "quick_ratio": 5.02,
    "debt_to_equity": 5.03,
    "lt_debt_to_equity": 5.04,
    "insider_ownership": 6.01,
    "insider_transactions": 6.02,
    "institutional_ownership": 6.03,
    "institutional_transactions": 6.04,
    "short_float": 6.06,
    "short_ratio": 6.07,
    "short_interest": 6_000_008,
    "beta": 7.01,
    "rsi_14": 7.02,
    "atr_14": 7.03,
    "sma_20": 7.04,
    "sma_50": 7.05,
    "sma_200": 7.06,
    "volatility_week": 7.07,
    "volatility_month": 7.08,
    "perf_week": 8.01,
    "perf_month": 8.02,
    "perf_quarter": 8.03,
    "perf_half_year": 8.04,
    "perf_year": 8.05,
    "perf_ytd": 8.06,
    "dividend_ttm": 9.01,
    "dividend_yield": 9.02,
    "payout_ratio": 9.03,
    "week_52_high": 10.01,
    "week_52_high_distance": 10.02,
    "week_52_low": 10.03,
    "week_52_low_distance": 10.04,
    "sector": "Technology",
    "industry": "Semiconductors",
    "country": "USA",
    "employees": 12_345,
    "ipo_date": date(2010, 6, 29),
    "description_yfinance": "yfinance description",
    "description_finviz": "finviz description",
    "recommendation": 1.8,
    "eps_5yr_cagr": 11.01,
    "eps_q1_yoy": 11.02,
    "eps_q2_yoy": 11.03,
    "eps_raw_score": 11.04,
    "eps_rating": 88,
    "eps_years_available": 5,
    "data_source": "finviz",
    "data_source_timestamp": None,
    "finviz_snapshot_revision": "rev-1",
    "finviz_snapshot_at": datetime(2026, 10, 18, 20, 0),
    "yahoo_profile_refreshed_at": datetime(2026, 10, 17, 21, 0),
    "yahoo_statements_refreshed_at": datetime(2026, 10, 16, 22, 0),
    "technicals_refreshed_at": datetime(2026, 10, 15, 23, 0),
    "field_completeness_score": 77,
    "field_provenance": {"market_cap": "finviz"},
    "market_cap_usd": 1_000_000_012,
    "adv_usd": 13_000_013,
    "fx_metadata": {"from_currency": "USD", "to_currency": "USD", "rate": 1.0},
}

READ_AFTER_INSERT = {
    "market_cap": 1_000_000_001,
    "shares_outstanding": 2_000_000_002,
    "avg_volume": 3_000_003,
    "relative_volume": 1.04,
    "pe_ratio": 2.05,
    "forward_pe": 2.06,
    "peg_ratio": 2.07,
    "price_to_book": 2.08,
    "price_to_sales": 2.09,
    "price_to_cash": 2.10,
    "price_to_fcf": 2.11,
    "ev_ebitda": 2.12,
    "ev_sales": 2.13,
    "target_price": 2.14,
    "eps_current": 3.01,
    "eps_next_y": 3.02,
    "eps_next_5y": 3.03,
    "eps_next_q": 3.04,
    "eps_growth_quarterly": 3.05,
    "eps_growth_annual": 3.07,
    "eps_growth_yy": 3.07,
    "revenue_current": 4_000_000_009,
    "revenue_growth": 3.10,
    "sales_past_5y": 3.11,
    "sales_growth_yy": 3.12,
    "sales_growth_qq": 3.13,
    "recent_quarter_date": "2026-Q2",
    "previous_quarter_date": "2026-Q1",
    "growth_reporting_cadence": "quarterly",
    "growth_metric_basis": "quarterly_qoq",
    "growth_comparable_period_date": "2025-06-30",
    "growth_reference_gap_days": 91,
    "next_earnings_date": date(2026, 11, 5),
    "event_calendar_as_of_date": date(2026, 10, 19),
    "recent_earnings": HISTORY,
    "eps_growth_qq": 3.05,
    "profit_margin": 4.01,
    "operating_margin": 4.02,
    "gross_margin": 4.03,
    "roe": 4.04,
    "roa": 4.05,
    "roic": 4.06,
    "current_ratio": 5.01,
    "quick_ratio": 5.02,
    "debt_to_equity": 5.03,
    "lt_debt_to_equity": 5.04,
    "insider_ownership": 6.01,
    "insider_transactions": 6.02,
    "institutional_ownership": 6.03,
    "institutional_transactions": 6.04,
    "short_float": 6.06,
    "short_ratio": 6.07,
    "short_interest": 6_000_008,
    "beta": 7.01,
    "rsi_14": 7.02,
    "atr_14": 7.03,
    "sma_20": 7.04,
    "sma_50": 7.05,
    "sma_200": 7.06,
    "volatility_week": 7.07,
    "volatility_month": 7.08,
    "perf_week": 8.01,
    "perf_month": 8.02,
    "perf_quarter": 8.03,
    "perf_half_year": 8.04,
    "perf_year": 8.05,
    "perf_ytd": 8.06,
    "dividend_ttm": 9.01,
    "dividend_yield": 9.02,
    "payout_ratio": 9.03,
    "week_52_high": 10.01,
    "week_52_high_distance": 10.02,
    "week_52_low": 10.03,
    "week_52_low_distance": 10.04,
    "sector": "Technology",
    "industry": "Semiconductors",
    "country": "USA",
    "employees": 12_345,
    "ipo_date": date(2010, 6, 29),
    "first_trade_date": int(datetime(2010, 6, 29).timestamp()),
    "description_yfinance": "yfinance description",
    "description_finviz": "finviz description",
    "recommendation": 1.8,
    "eps_5yr_cagr": 11.01,
    "eps_q1_yoy": 11.02,
    "eps_q2_yoy": 11.03,
    "eps_raw_score": 11.04,
    "eps_rating": 88,
    "eps_years_available": 5,
    "data_source": "finviz",
    "finviz_snapshot_revision": "rev-1",
    "finviz_snapshot_at": "2026-10-18T20:00:00",
    "yahoo_profile_refreshed_at": "2026-10-17T21:00:00",
    "yahoo_statements_refreshed_at": "2026-10-16T22:00:00",
    "technicals_refreshed_at": "2026-10-15T23:00:00",
    "field_completeness_score": 77,
    "field_provenance": {"market_cap": "finviz"},
    "market_cap_usd": 1_000_000_012,
    "adv_usd": 13_000_013,
    "fx_metadata": {"from_currency": "USD", "to_currency": "USD", "rate": 1.0},
    "description": "finviz description",
}


def _ownership_availability(status, reason_code):
    return {
        "institutional_ownership": {"status": status, "reason_code": reason_code, "support_state": "partial"},
        "insider_ownership": {"status": status, "reason_code": reason_code, "support_state": "partial"},
        "short_interest": {"status": status, "reason_code": reason_code, "support_state": "supported"},
    }


def _service():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return FundamentalsCacheService(redis_client=None, session_factory=factory), factory


def _row(factory):
    db = factory()
    try:
        return db.query(StockFundamental).filter_by(symbol="AAPL").one()
    finally:
        db.close()


def _columns(factory):
    row = _row(factory)
    return {
        column.name: getattr(row, column.name)
        for column in StockFundamental.__table__.columns
        if column.name not in {"id", "updated_at"}
    }


def _insert(service):
    assert service._store_in_database("AAPL", dict(FULL_PAYLOAD), data_source="finviz", market="US") is True


def test_insert_writes_every_mapped_column():
    service, factory = _service()

    _insert(service)

    assert _columns(factory) == COLUMNS_AFTER_INSERT


def test_insert_ignores_the_guards_an_update_applies():
    service, factory = _service()
    payload = {"pe_ratio": 1.0, "next_earnings_date": "2026-11-05"}

    assert service._store_in_database("AAPL", payload, data_source="finviz", market="US") is True

    db = factory()
    try:
        stored = db.execute(text("SELECT quote(fx_metadata), quote(market_cap) FROM stock_fundamentals")).one()
    finally:
        db.close()
    assert tuple(stored) == ("'null'", "NULL")
    row = _row(factory)
    assert (row.next_earnings_date, row.event_calendar_as_of_date) == (date(2026, 11, 5), None)


def test_single_and_bulk_reads_map_every_column():
    service, factory = _service()
    _insert(service)
    updated_at = _row(factory).updated_at

    single = service._get_from_database("AAPL")
    many = service._get_many_from_database(["AAPL", "MISSING"])

    assert single == (
        {**READ_AFTER_INSERT, "field_availability": _ownership_availability("available", None)},
        updated_at,
    )
    assert many == {"AAPL": (READ_AFTER_INSERT, updated_at), "MISSING": (None, None)}


def test_updates_keep_guarded_columns_and_clear_the_rest():
    service, factory = _service()
    _insert(service)

    assert service._store_in_database("AAPL", dict(PARTIAL_UPDATE), data_source="yfinance", market="US") is True

    assert _columns(factory) == {
        **dict.fromkeys(COLUMNS_AFTER_INSERT),
        "symbol": "AAPL",
        "market_cap": 1_000_000_001,
        "shares_outstanding": 2_000_000_002,
        "avg_volume": 0,
        "pe_ratio": 20.5,
        "eps_growth_quarterly": 30.5,
        "next_earnings_date": date(2026, 11, 5),
        "event_calendar_as_of_date": date(2026, 10, 19),
        "recent_earnings": HISTORY,
        "sector": "Healthcare",
        "ipo_date": date(2001, 2, 3),
        "recommendation": 2.5,
        "data_source": "yfinance",
        "technicals_refreshed_at": datetime(2026, 10, 20, 1, 2, 3),
        "field_completeness_score": 41,
        "field_provenance": {"pe_ratio": "yfinance"},
        "market_cap_usd": 1_000_000_012,
        "adv_usd": 13_000_013,
        "fx_metadata": {"rate": 0.5},
    }
    fundamentals, _ = service._get_from_database("AAPL")
    assert fundamentals == {
        **dict.fromkeys(READ_AFTER_INSERT),
        "market_cap": 1_000_000_001,
        "shares_outstanding": 2_000_000_002,
        "avg_volume": 0,
        "pe_ratio": 20.5,
        "eps_growth_quarterly": 30.5,
        "eps_growth_qq": 30.5,
        "next_earnings_date": date(2026, 11, 5),
        "event_calendar_as_of_date": date(2026, 10, 19),
        "recent_earnings": HISTORY,
        "sector": "Healthcare",
        "ipo_date": date(2001, 2, 3),
        "first_trade_date": int(datetime(2001, 2, 3).timestamp()),
        "recommendation": 2.5,
        "data_source": "yfinance",
        "technicals_refreshed_at": "2026-10-20T01:02:03",
        "field_completeness_score": 41,
        "field_provenance": {"pe_ratio": "yfinance"},
        "market_cap_usd": 1_000_000_012,
        "adv_usd": 13_000_013,
        "fx_metadata": {"rate": 0.5},
        "field_availability": _ownership_availability("missing", "missing_supported_field_value"),
    }

    assert service._store_in_database("AAPL", dict(CALENDAR_UPDATE), data_source="yfinance", market="US") is True

    assert _columns(factory) == {
        **dict.fromkeys(COLUMNS_AFTER_INSERT),
        "symbol": "AAPL",
        "market_cap": 5,
        "shares_outstanding": 2_000_000_002,
        "avg_volume": 0,
        "event_calendar_as_of_date": date(2026, 10, 20),
        "recent_earnings": HISTORY,
        "data_source": "yfinance",
        "field_completeness_score": 5,
        "field_provenance": {},
        "market_cap_usd": 1_000_000_012,
        "adv_usd": 13_000_013,
        "fx_metadata": {"rate": 0.5},
    }


def test_storing_an_unchanged_payload_still_moves_updated_at():
    service, factory = _service()
    _insert(service)
    stale = datetime(2000, 1, 1)
    db = factory()
    db.query(StockFundamental).filter_by(symbol="AAPL").update({"updated_at": stale})
    db.commit()
    db.close()

    _insert(service)

    assert _row(factory).updated_at > stale


@pytest.mark.parametrize("finviz_fields", [{"description_finviz": ""}, {}])
def test_description_falls_back_to_yfinance(finviz_fields):
    service, _ = _service()
    payload = {"description_yfinance": "yfinance description", **finviz_fields}

    assert service._store_in_database("AAPL", payload, data_source="yfinance", market="US") is True

    fundamentals, _ = service._get_from_database("AAPL")
    assert fundamentals["description"] == "yfinance description"


def test_insert_derives_ipo_date_from_first_trade_date_ms():
    service, factory = _service()

    assert service._store_in_database(
        "AAPL", {"first_trade_date_ms": 981_201_600_000}, data_source="yfinance", market="US"
    ) is True

    assert _row(factory).ipo_date == date(2001, 2, 3)
    fundamentals, _ = service._get_from_database("AAPL")
    assert fundamentals["first_trade_date"] == int(datetime(2001, 2, 3).timestamp())
