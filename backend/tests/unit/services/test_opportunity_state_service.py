from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from app.analysis.patterns.config import SetupEngineParameters
from app.scanners.base_screener import StockData
from app.services.opportunity_state_service import (
    build_data_limited_projection,
    build_opportunity_projection,
)


@pytest.mark.parametrize("market,volume,available,passed", [
    ("US", 99_999_999, True, False),
    ("US", 100_000_000, True, True),
    (" hk ", 8_000_000, True, True),
    ("HK", 7_999_999, True, False),
    ("CN", "7000000", True, True),
    ("JP", 149_999_999, True, False),
    ("TW", 30_000_000, True, True),
    ("UNKNOWN", 1_000_000_000, False, None),
    (None, 1_000_000_000, False, None),
    ("US", None, False, None),
    ("US", True, False, None),
    ("US", float("inf"), False, None),
    ("US", float("nan"), False, None),
    ("US", "unavailable", False, None),
])
def test_public_liquidity_evidence_preserves_market_floor_and_missing_values(
    market, volume, available, passed,
):
    """Catch shared Social liquidity drifting from local floors or treating missing as fail."""
    from app.services import opportunity_state_service
    adapter = getattr(opportunity_state_service, "read_liquidity_evidence", None)
    assert callable(adapter), "public liquidity evidence adapter is missing"
    evidence = adapter(market, volume)
    assert evidence.available is available
    assert evidence.value is passed


@pytest.fixture
def stock_data() -> StockData:
    dates = pd.DatetimeIndex(["2026-08-20", "2026-08-21"])
    price_data = pd.DataFrame(
        {"Close": [99.0, 100.0], "Volume": [1_500_000, 1_500_000]},
        index=dates,
    )
    benchmark_data = pd.DataFrame(
        {"Close": [499.0, 500.0], "Volume": [1_000_000, 1_000_000]},
        index=dates,
    )
    return StockData(
        symbol="TEST",
        price_data=price_data,
        benchmark_data=benchmark_data,
        fundamentals={},
        next_earnings_date=None,
        event_calendar_available=True,
        market="US",
        exchange="NASDAQ",
        benchmark_symbol="SPY",
    )


@pytest.fixture
def result() -> dict[str, object]:
    return {
        "avg_dollar_volume": 150_000_000,
        "data_status": "complete",
        "is_scannable": True,
        "rs_rating_1m": 90.0,
        "rs_rating_3m": 80.0,
        "stage": 2,
        "ma_alignment": True,
        "setup_engine": {
            "pattern_primary": "vcp",
            "bb_squeeze": True,
            "tight_closes_count": 3,
            "quiet_days_10d": 3,
            "volume_vs_50d": 0.7,
            "rs_vs_spy_65d": 0.08,
            "rs_line_new_high": True,
            "rs_line_blue_dot": False,
            "setup_ready": True,
            "in_early_zone": True,
            "extended_from_pivot": False,
            "explain": {"invalidation_flags": []},
        },
    }


def test_build_projection_uses_market_benchmark_and_local_liquidity(
    stock_data: StockData,
    result: dict[str, object],
):
    """Break caught: applying a USD floor or losing point-in-time HK identity."""
    stock_data.market = "HK"
    stock_data.exchange = "HKEX"
    stock_data.benchmark_symbol = "^HSI"
    result["avg_dollar_volume"] = 9_000_000

    projection = build_opportunity_projection(
        result, stock_data, SetupEngineParameters()
    )
    evidence = projection["opportunity_state"]

    assert set(projection) == {
        "correction_survivor",
        "resilience_score",
        "action_state",
        "opportunity_state",
    }
    assert evidence["market"] == "HK"
    assert evidence["mic"] == "XHKG"
    assert evidence["as_of_date"] == "2026-08-21"
    assert evidence["benchmark_symbol"] == "^HSI"
    assert evidence["benchmark_as_of_date"] == "2026-08-21"
    assert evidence["metrics"]["liquidity_floor_local"] == 8_000_000
    assert evidence["metrics"]["liquidity_passes"] is True


def test_missing_event_key_is_data_limited(
    stock_data: StockData,
    result: dict[str, object],
):
    """Break caught: treating an unavailable calendar as known no-event evidence."""
    stock_data.event_calendar_available = False

    projection = build_opportunity_projection(
        result, stock_data, SetupEngineParameters()
    )

    assert projection["action_state"] == "data_limited"
    assert projection["opportunity_state"]["data_availability"]["event_calendar"] == "unavailable"


def test_present_null_event_is_available_and_does_not_invent_event_risk(
    stock_data: StockData,
    result: dict[str, object],
):
    """Break caught: conflating an available calendar with no event and a missing calendar."""
    stock_data.fundamentals = {}
    stock_data.event_calendar_available = True
    stock_data.next_earnings_date = None

    projection = build_opportunity_projection(
        result, stock_data, SetupEngineParameters()
    )

    assert projection["action_state"] == "setup_ready"
    assert projection["opportunity_state"]["data_availability"]["event_calendar"] == "available"


def test_event_inside_window_has_event_risk_precedence(
    stock_data: StockData,
    result: dict[str, object],
):
    """Break caught: failing to compare the normalized event against the row as-of date."""
    stock_data.next_earnings_date = date(2026, 8, 28)

    projection = build_opportunity_projection(
        result,
        stock_data,
        SetupEngineParameters(earnings_soon_window_days=7),
    )

    assert projection["action_state"] == "event_risk"
    assert projection["opportunity_state"]["action_reasons"] == ["earnings_soon"]


def test_hard_setup_flag_drives_exit_risk_from_is_hard_boolean(
    stock_data: StockData,
    result: dict[str, object],
):
    """Break caught: inferring hard risk from severity or code instead of Task 2 evidence."""
    result["setup_engine"]["explain"]["invalidation_flags"] = [
        {
            "code": "breaks_50d_support",
            "message": "support failed",
            "severity": "medium",
            "is_hard": True,
        }
    ]

    projection = build_opportunity_projection(
        result, stock_data, SetupEngineParameters()
    )

    assert projection["action_state"] == "exit_risk"
    assert projection["opportunity_state"]["metrics"]["hard_invalidation"] is True


def test_current_scan_does_not_invent_prior_run_deterioration(
    stock_data: StockData,
    result: dict[str, object],
):
    """Break caught: requiring or synthesizing prior-run evidence during current scan assembly."""
    projection = build_opportunity_projection(
        result, stock_data, SetupEngineParameters()
    )

    assert projection["action_state"] == "setup_ready"
    assert projection["opportunity_state"]["data_availability"]["prior_run"] == "not_requested"


def test_present_null_primary_pattern_with_alternate_pass_has_numeric_score(
    stock_data: StockData,
    result: dict[str, object],
):
    """Break caught: the assembler collapsing a valid no-pattern result into unknown evidence."""
    result["setup_engine"]["pattern_primary"] = None

    projection = build_opportunity_projection(
        result, stock_data, SetupEngineParameters()
    )

    assert projection["correction_survivor"] is True
    assert projection["resilience_score"] == 89.0
    assert projection["action_state"] == "setup_ready"
    assert projection["opportunity_state"]["metrics"]["pattern_primary"] is None


def test_present_null_primary_pattern_with_failed_alternates_is_watch(
    stock_data: StockData,
    result: dict[str, object],
):
    """Break caught: known absent structure being reported as unavailable evidence."""
    setup = result["setup_engine"]
    setup.update(
        {
            "pattern_primary": None,
            "bb_squeeze": False,
            "tight_closes_count": 2,
            "quiet_days_10d": 2,
            "volume_vs_50d": 0.81,
        }
    )

    projection = build_opportunity_projection(
        result, stock_data, SetupEngineParameters()
    )

    assert projection["correction_survivor"] is False
    assert projection["resilience_score"] == 77.0
    assert projection["action_state"] == "watch"
    assert "structure_gate" in projection["opportunity_state"]["failed_checks"]
    assert projection["opportunity_state"]["metrics"]["pattern_primary"] is None


@pytest.mark.parametrize("pattern_value", [pytest.param(None, id="missing"), pytest.param([], id="malformed")])
def test_missing_or_malformed_primary_pattern_remains_unknown(
    stock_data: StockData,
    result: dict[str, object],
    pattern_value,
):
    """Break caught: absent or malformed pattern evidence becoming a known false score input."""
    setup = result["setup_engine"]
    if pattern_value is None:
        setup.pop("pattern_primary")
    else:
        setup["pattern_primary"] = pattern_value
    setup.update(
        {
            "bb_squeeze": False,
            "tight_closes_count": 2,
            "quiet_days_10d": 2,
            "volume_vs_50d": 0.81,
        }
    )

    projection = build_opportunity_projection(
        result, stock_data, SetupEngineParameters()
    )

    assert projection["correction_survivor"] is False
    assert projection["resilience_score"] is None
    assert projection["action_state"] == "data_limited"
    assert "structure_gate" not in projection["opportunity_state"]["failed_checks"]
    assert projection["opportunity_state"]["metrics"]["pattern_primary"] is None


def test_data_limited_projection_preserves_identity_and_safe_reason(
    stock_data: StockData,
    result: dict[str, object],
):
    """Break caught: returning an unstructured fallback or leaking exception text."""
    stock_data.next_earnings_date = date(2026, 8, 21)
    projection = build_data_limited_projection(
        result,
        stock_data,
        "opportunity_policy_error",
    )

    assert projection["correction_survivor"] is False
    assert projection["resilience_score"] is None
    assert projection["action_state"] == "data_limited"
    assert projection["opportunity_state"]["market"] == "US"
    assert projection["opportunity_state"]["mic"] == "XNAS"
    assert projection["opportunity_state"]["as_of_date"] == "2026-08-21"
    assert projection["opportunity_state"]["benchmark_symbol"] == "SPY"
    assert projection["opportunity_state"]["benchmark_as_of_date"] == "2026-08-21"
    assert projection["opportunity_state"]["action_reasons"] == [
        "opportunity_policy_error"
    ]
