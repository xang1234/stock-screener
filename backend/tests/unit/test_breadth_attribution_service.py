"""Unit tests for ``BreadthAttributionService``."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.services.breadth_attribution_service import (
    NO_GROUP_LABEL,
    BreadthAttributionService,
)


def _frame(closes: list[float], start: str = "2026-05-01") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=idx)


def test_compute_returns_empty_when_no_symbols():
    service = BreadthAttributionService()
    result = service.compute(
        symbols_meta=[],
        price_data={},
        target_dates=[date(2026, 5, 5)],
    )
    assert result == []


def test_compute_attributes_up_mover_to_ibd_group():
    service = BreadthAttributionService()
    # Day 1 = 100, Day 2 = 105 → +5%, exceeds 4% threshold.
    price_data = {"PLTR": _frame([100.0, 105.0])}
    symbols_meta = [
        {
            "symbol": "PLTR",
            "company_name": "Palantir Technologies",
            "ibd_industry_group": "Computer Software-Database",
        }
    ]

    result = service.compute(
        symbols_meta=symbols_meta,
        price_data=price_data,
        target_dates=[date(2026, 5, 2)],
    )

    assert len(result) == 1
    day = result[0]
    assert day["date"] == "2026-05-02"
    assert day["stocks_up_4pct"] == 1
    assert day["stocks_down_4pct"] == 0
    assert len(day["groups"]) == 1
    group = day["groups"][0]
    assert group["group"] == "Computer Software-Database"
    assert group["up_count"] == 1
    assert group["down_count"] == 0
    assert group["net"] == 1
    assert group["up_stocks"][0]["symbol"] == "PLTR"
    assert group["up_stocks"][0]["name"] == "Palantir Technologies"
    assert group["up_stocks"][0]["pct_change"] == pytest.approx(5.0)
    assert group["up_stocks"][0]["close"] == pytest.approx(105.0)


def test_compute_attributes_down_mover():
    service = BreadthAttributionService()
    price_data = {"XYZ": _frame([100.0, 94.0])}  # -6%

    result = service.compute(
        symbols_meta=[{"symbol": "XYZ", "ibd_industry_group": "Banks-Money Center"}],
        price_data=price_data,
        target_dates=[date(2026, 5, 2)],
    )

    day = result[0]
    assert day["stocks_up_4pct"] == 0
    assert day["stocks_down_4pct"] == 1
    group = day["groups"][0]
    assert group["down_count"] == 1
    assert group["net"] == -1
    assert group["down_stocks"][0]["pct_change"] == pytest.approx(-6.0)


def test_compute_buckets_missing_group_into_no_group():
    service = BreadthAttributionService()
    price_data = {
        "AAA": _frame([100.0, 106.0]),  # +6%
        "BBB": _frame([100.0, 108.0]),  # +8%
    }
    symbols_meta = [
        {"symbol": "AAA"},  # No ibd_industry_group key
        {"symbol": "BBB", "ibd_industry_group": "  "},  # Whitespace → No Group
    ]

    result = service.compute(
        symbols_meta=symbols_meta,
        price_data=price_data,
        target_dates=[date(2026, 5, 2)],
    )

    day = result[0]
    assert day["stocks_up_4pct"] == 2
    assert len(day["groups"]) == 1
    no_group = day["groups"][0]
    assert no_group["group"] == NO_GROUP_LABEL
    assert no_group["up_count"] == 2
    assert {entry["symbol"] for entry in no_group["up_stocks"]} == {"AAA", "BBB"}


def test_compute_skips_movers_below_threshold():
    service = BreadthAttributionService()
    # +3.5% — below the 4% threshold, should not appear.
    price_data = {"FLAT": _frame([100.0, 103.5])}
    result = service.compute(
        symbols_meta=[{"symbol": "FLAT", "ibd_industry_group": "Retail"}],
        price_data=price_data,
        target_dates=[date(2026, 5, 2)],
    )
    assert result[0]["groups"] == []
    assert result[0]["stocks_up_4pct"] == 0


def test_compute_sorts_groups_by_total_activity_then_net():
    service = BreadthAttributionService()
    # Group A: 1 up, 0 down (activity=1, net=1)
    # Group B: 2 up, 1 down (activity=3, net=1)
    # Group C: 0 up, 2 down (activity=2, net=-2)
    price_data = {
        "A1": _frame([100.0, 110.0]),
        "B1": _frame([100.0, 110.0]),
        "B2": _frame([100.0, 110.0]),
        "B3": _frame([100.0, 90.0]),
        "C1": _frame([100.0, 90.0]),
        "C2": _frame([100.0, 90.0]),
    }
    meta = [
        {"symbol": "A1", "ibd_industry_group": "Alpha"},
        {"symbol": "B1", "ibd_industry_group": "Bravo"},
        {"symbol": "B2", "ibd_industry_group": "Bravo"},
        {"symbol": "B3", "ibd_industry_group": "Bravo"},
        {"symbol": "C1", "ibd_industry_group": "Charlie"},
        {"symbol": "C2", "ibd_industry_group": "Charlie"},
    ]
    result = service.compute(
        symbols_meta=meta,
        price_data=price_data,
        target_dates=[date(2026, 5, 2)],
    )
    groups = [g["group"] for g in result[0]["groups"]]
    # Bravo first (highest activity=3), then Charlie (activity=2), then Alpha.
    assert groups == ["Bravo", "Charlie", "Alpha"]


def test_compute_returns_history_oldest_to_newest():
    service = BreadthAttributionService()
    # Three-day frame where day 2 has a +5% move and day 3 has -5%.
    price_data = {"X": _frame([100.0, 105.0, 99.0])}
    result = service.compute(
        symbols_meta=[{"symbol": "X", "ibd_industry_group": "G"}],
        price_data=price_data,
        target_dates=[date(2026, 5, 2), date(2026, 5, 3)],
    )
    assert [day["date"] for day in result] == ["2026-05-02", "2026-05-03"]
    assert result[0]["stocks_up_4pct"] == 1
    assert result[1]["stocks_down_4pct"] == 1


def test_compute_skips_dates_with_missing_price_data():
    service = BreadthAttributionService()
    price_data = {"X": _frame([100.0, 105.0])}  # Only covers 2026-05-01, 2026-05-02
    result = service.compute(
        symbols_meta=[{"symbol": "X", "ibd_industry_group": "G"}],
        price_data=price_data,
        target_dates=[date(2026, 5, 2), date(2026, 5, 10)],
    )
    # 2026-05-10 has no price → no group entry for that day.
    by_date = {row["date"]: row for row in result}
    assert by_date["2026-05-02"]["stocks_up_4pct"] == 1
    assert by_date["2026-05-10"]["groups"] == []
    assert by_date["2026-05-10"]["stocks_up_4pct"] == 0


def test_compute_skips_symbols_without_price_data():
    service = BreadthAttributionService()
    result = service.compute(
        symbols_meta=[{"symbol": "MISSING", "ibd_industry_group": "G"}],
        price_data={"MISSING": None},
        target_dates=[date(2026, 5, 2)],
    )
    assert result[0]["groups"] == []
