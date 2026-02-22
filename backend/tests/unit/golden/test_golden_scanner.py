"""Golden regression tests for full SetupEngineScanner pipeline (SE-G7).

11 curated cases exercising ``SetupEngineScanner.scan_stock()`` end-to-end
through all 5 phases: data prep → detector aggregation → readiness
computation → gate evaluation → rating & packaging.

Cases pin expected ``ScreenerResult`` outputs — rating, score ranges,
gate checks, readiness fields, and the complete ``setup_engine`` payload.

Snapshot review/approval process
--------------------------------
When behavior changes require snapshot regeneration:

1. Run ``pytest tests/unit/golden/test_golden_scanner.py --golden-update``
2. Review the diff in ``snapshots/scanner_*.json``
3. Update inline expectation ranges if actual values shifted
4. Commit with message body explaining the change rationale::

    test(scanner): update golden snapshots after [description]

    Rationale: [why behavior changed]
    Affected cases: [list of case IDs whose snapshots changed]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.scanners.setup_engine_screener import SetupEngineScanner

from .conftest import (
    assert_golden_scanner_match,
    golden_benchmark_frame,
    golden_ohlcv_frame,
    golden_stock_data,
    maybe_export_snapshot,
    _expand_weekly_to_daily,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SCANNER = SetupEngineScanner()

# Common readiness fields that should be non-null when a pattern is detected
# and benchmark data is present
_STANDARD_READINESS_FIELDS = [
    "distance_to_pivot_pct",
    "atr14_pct",
    "bb_width_pct",
    "volume_vs_50d",
    "rs",
]


# ---------------------------------------------------------------------------
# Positive pattern detection cases (6)
# ---------------------------------------------------------------------------


def _build_scanner_3wt_strict() -> tuple[dict[str, Any], dict[str, Any]]:
    """Scanner case: 3WT strict detection through full pipeline.

    350 daily bars: 300-bar uptrend 50→95, then 50-bar tight zone with
    Friday closes within ±0.3% of 100.
    """
    rng = np.random.default_rng(42)
    n = 350
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)

    # Uptrend: 50 → 95 over 300 bars
    uptrend = np.linspace(50, 95, 300)
    # Tight zone: 50 bars oscillating near 100 with ±0.3% jitter
    tight = 100.0 + rng.uniform(-0.3, 0.3, 50)
    close = np.concatenate([uptrend, tight])

    price_df = golden_ohlcv_frame(index=index, close=close)
    bench_df = golden_benchmark_frame(index=index)
    data = golden_stock_data(symbol="GOLDEN_SCAN_3WT", price_data=price_df, benchmark_data=bench_df)

    fixture = {"symbol": "GOLDEN_SCAN_3WT", "data": data}
    expectation = {
        # setup_ready=False because derived_ready merges all detector failures
        # (other detectors' not_detected checks), even though all 8 gates pass
        "passes": False,
        "screener_name": "setup_engine",
        # Actual: score=92.5, quality=100, readiness=81.3
        "score_range": (78, 100),
        "rating": "Watch",
        "pattern_primary": "three_weeks_tight",
        "pivot_type": "tight_area_high",
        "setup_score": (78, 100),
        "quality_score": (85, 100),
        "readiness_score": (69, 95),
        "required_readiness_fields": _STANDARD_READINESS_FIELDS,
        # All 8 gates pass despite setup_ready=False
        "passed_contains": [
            "setup_score_ok", "quality_floor_ok", "readiness_floor_ok",
            "in_early_zone", "atr14_within_limit", "volume_sufficient",
            "rs_leadership_ok", "stage_ok",
        ],
    }
    return fixture, expectation


def _build_scanner_htf_classic() -> tuple[dict[str, Any], dict[str, Any]]:
    """Scanner case: High-Tight Flag detection through full pipeline.

    350 bars: 130-bar base 40→55, 30-bar pole to 115 (>100% return),
    16-bar flag 113→110, rest at 112. High volume on pole, low on flag.
    """
    n = 350
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)

    base = np.linspace(40, 55, 130)
    pole = np.linspace(55, 115, 30)
    flag = np.linspace(113, 110, 16)
    rest = np.full(n - 130 - 30 - 16, 112.0)
    close = np.concatenate([base, pole, flag, rest])

    # Volume: high on pole, low on flag
    volume = np.full(n, 1_000_000.0)
    volume[130:160] = 3_000_000  # Pole: high volume
    volume[160:176] = 500_000    # Flag: low volume

    price_df = golden_ohlcv_frame(index=index, close=close, volume=volume)
    bench_df = golden_benchmark_frame(index=index)
    data = golden_stock_data(symbol="GOLDEN_SCAN_HTF", price_data=price_df, benchmark_data=bench_df)

    fixture = {"symbol": "GOLDEN_SCAN_HTF", "data": data}
    expectation = {
        # 3WT wins primary selection due to higher rank score,
        # but HTF is present as a candidate (verified below)
        "screener_name": "setup_engine",
        # Actual: score=95.6, quality=100, readiness=88.9
        "score_range": (81, 100),
        "rating": "Watch",
        "passes": False,
        "pattern_primary": "three_weeks_tight",
        "setup_score": (81, 100),
        "quality_score": (85, 100),
        # HTF detector fires — verified by passed_contains
        "passed_contains": ["pole_candidates_found", "flag_candidates_validated"],
        "required_readiness_fields": _STANDARD_READINESS_FIELDS,
    }
    return fixture, expectation


def _build_scanner_nr7_inside_day() -> tuple[dict[str, Any], dict[str, Any]]:
    """Scanner case: NR7+Inside Day detection through full pipeline.

    350 bars: uptrend with NR7+inside day on the last bar.
    Bar 349 has the narrowest range in 7 bars and fits inside bar 348.
    """
    n = 350
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)

    close = np.linspace(60, 120, n)

    # Build high/low with progressively narrower ranges near the end
    high = close * 1.015
    low = close * 0.985

    # Create expanding ranges for bars 340-348, then narrow for bar 349
    for i in range(340, 349):
        spread = 0.5 + (i - 340) * 0.3
        high[i] = close[i] + spread
        low[i] = close[i] - spread

    # Bar 348: wide parent bar
    high[348] = close[348] + 3.0
    low[348] = close[348] - 3.0

    # Bar 349: narrowest in 7 bars AND inside bar 348
    high[349] = close[349] + 0.15
    low[349] = close[349] - 0.15

    # Ensure bar 349 fits inside bar 348
    high[349] = min(high[349], high[348] - 0.01)
    low[349] = max(low[349], low[348] + 0.01)

    price_df = golden_ohlcv_frame(index=index, close=close, high=high, low=low)
    bench_df = golden_benchmark_frame(index=index)
    data = golden_stock_data(symbol="GOLDEN_SCAN_NR7", price_data=price_df, benchmark_data=bench_df)

    fixture = {"symbol": "GOLDEN_SCAN_NR7", "data": data}
    expectation = {
        "screener_name": "setup_engine",
        # Actual: score=89.6, quality=100, readiness=73.9
        "score_range": (76, 100),
        "rating": "Watch",
        "pattern_primary": "nr7_inside_day",
        "pivot_type_contains": "trigger_high",
        "setup_score": (76, 100),
        "quality_score": (85, 100),
        "readiness_score": (63, 85),
        "required_readiness_fields": _STANDARD_READINESS_FIELDS,
    }
    return fixture, expectation


def _build_scanner_cwh_classic() -> tuple[dict[str, Any], dict[str, Any]]:
    """Scanner case: Cup-with-Handle detection through full pipeline.

    Uses the weekly-first, daily-expansion strategy. The weekly shape
    is the same proven shape from SE-G1's ``_build_cup_with_handle_classic()``,
    expanded to daily bars via ``_expand_weekly_to_daily()``.
    """
    # Step 1: Build the proven weekly cup shape (55 weekly bars)
    weekly_index = pd.date_range("2022-06-03", periods=55, freq="W-FRI")

    weekly_close = np.concatenate([
        np.array([88, 92, 96, 99]),            # Weeks 0-3: lead-in
        np.array([103]),                         # Week 4: left lip peak
        np.array([99, 95]),                      # Weeks 5-6: drop-off
        np.linspace(92, 72, 11),                 # Weeks 7-17: descent to cup low
        np.linspace(74, 97, 11),                 # Weeks 18-28: recovery
        np.array([101]),                          # Week 29: right lip peak
        np.array([99, 98]),                       # Weeks 30-31: handle start
        np.array([97, 98]),                       # Weeks 32-33: handle body
        np.array([99, 100]),                      # Weeks 34-35: handle exit
        np.linspace(100, 102, 19),               # Weeks 36-54: continuation
    ])

    # Override highs at swing peaks for detect_swings(left=2, right=2)
    weekly_high = weekly_close * 1.01
    weekly_high[4] = 106.0    # Left lip
    weekly_high[29] = 104.0   # Right lip

    # Volume: declining through cup and handle
    weekly_volume = np.empty(55)
    weekly_volume[:5] = 1_500_000
    weekly_volume[5:18] = np.linspace(1_400_000, 800_000, 13)
    weekly_volume[18:29] = np.linspace(900_000, 1_000_000, 11)
    weekly_volume[29:36] = np.linspace(700_000, 500_000, 7)
    weekly_volume[36:] = np.linspace(600_000, 800_000, 19)

    weekly_df = golden_ohlcv_frame(
        index=weekly_index,
        close=weekly_close,
        high=weekly_high,
        volume=weekly_volume,
    )

    # Step 2: Expand weekly → daily
    daily_cup = _expand_weekly_to_daily(weekly_df)

    # Step 3: Prepend 75 daily bars of lead-in uptrend for warmup
    first_daily_date = daily_cup.index[0]
    leadin_end = first_daily_date - pd.Timedelta(days=1)
    leadin_index = pd.bdate_range(end=leadin_end, periods=75)
    leadin_close = np.linspace(60, float(daily_cup["Close"].iloc[0]) - 1, 75)
    leadin_df = golden_ohlcv_frame(index=leadin_index, close=leadin_close)

    # Combine
    price_df = pd.concat([leadin_df, daily_cup])

    bench_df = golden_benchmark_frame(index=price_df.index)
    data = golden_stock_data(symbol="GOLDEN_SCAN_CWH", price_data=price_df, benchmark_data=bench_df)

    fixture = {"symbol": "GOLDEN_SCAN_CWH", "data": data}
    expectation = {
        # 3WT wins primary selection; CWH detected as a candidate
        "screener_name": "setup_engine",
        # Actual: score=93.0, quality=100, readiness=82.5
        "score_range": (79, 100),
        "rating": "Watch",
        "passes": False,
        "pattern_primary": "three_weeks_tight",
        "setup_score": (79, 100),
        "quality_score": (85, 100),
        "readiness_score": (70, 95),
        # CWH detector fires — verified by passed_contains
        "passed_contains": ["cup_structure_candidates_found"],
        "required_readiness_fields": _STANDARD_READINESS_FIELDS,
    }
    return fixture, expectation


def _build_scanner_first_pullback() -> tuple[dict[str, Any], dict[str, Any]]:
    """Scanner case: First Pullback detection through full pipeline.

    350 bars: 250-bar uptrend 80→115, bars 250-265 pullback toward 21MA,
    bars 266-280 resumption bounce, bars 280-349 continuation.
    Volume: lower on pullback, higher on resumption.
    """
    n = 350
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)

    # Uptrend phase: 80 → 115 over 250 bars
    uptrend = np.linspace(80, 115, 250)

    # Compute approximate 21-day MA at end of uptrend
    # MA21 ≈ average of last 21 close values of uptrend
    ma21_approx = float(np.mean(uptrend[-21:]))

    # Pullback: 16 bars from 115 down toward MA21 (within ~1.5%)
    pullback_target = ma21_approx * 1.01  # Stay slightly above MA
    pullback = np.linspace(115, pullback_target, 16)

    # Resumption bounce: 15 bars from pullback low back up
    resumption = np.linspace(pullback_target, 116, 15)

    # Continuation: rest of the bars
    continuation_len = n - 250 - 16 - 15
    continuation = np.linspace(116, 120, continuation_len)

    close = np.concatenate([uptrend, pullback, resumption, continuation])

    # Volume pattern: lower on pullback, higher on resumption
    volume = np.full(n, 1_000_000.0)
    volume[250:266] = 600_000    # Pullback: dry up
    volume[266:281] = 1_500_000  # Resumption: higher volume

    price_df = golden_ohlcv_frame(index=index, close=close, volume=volume)
    bench_df = golden_benchmark_frame(index=index)
    data = golden_stock_data(
        symbol="GOLDEN_SCAN_FPB",
        price_data=price_df,
        benchmark_data=bench_df,
    )

    fixture = {"symbol": "GOLDEN_SCAN_FPB", "data": data}
    expectation = {
        # 3WT wins primary selection; first_pullback detected as a candidate
        "screener_name": "setup_engine",
        # Actual: score=89.8, quality=100, readiness=74.5
        "score_range": (76, 100),
        "rating": "Watch",
        "passes": False,
        "pattern_primary": "three_weeks_tight",
        "setup_score": (76, 100),
        "quality_score": (85, 100),
        "readiness_score": (63, 86),
        # First pullback detector fires
        "passed_contains": ["ma_tests_identified"],
        "required_readiness_fields": _STANDARD_READINESS_FIELDS,
    }
    return fixture, expectation


# ---------------------------------------------------------------------------
# Parametrized positive cases (5 non-VCP)
# ---------------------------------------------------------------------------

_POSITIVE_CASES = [
    ("scanner_3wt_strict", *_build_scanner_3wt_strict()),
    ("scanner_htf_classic", *_build_scanner_htf_classic()),
    ("scanner_nr7_inside_day", *_build_scanner_nr7_inside_day()),
    ("scanner_cwh_classic", *_build_scanner_cwh_classic()),
    ("scanner_first_pullback", *_build_scanner_first_pullback()),
]


@pytest.mark.parametrize(
    "case_id, fixture, expectation",
    _POSITIVE_CASES,
    ids=[c[0] for c in _POSITIVE_CASES],
)
def test_golden_scanner_positive(case_id, fixture, expectation, golden_update):
    """Golden positive scanner cases (non-VCP)."""
    result = _SCANNER.scan_stock(
        fixture["symbol"],
        fixture["data"],
    )
    maybe_export_snapshot(case_id, result, golden_update, is_scanner=True)
    assert_golden_scanner_match(result, expectation)


# ---------------------------------------------------------------------------
# VCP positive case (requires monkeypatch)
# ---------------------------------------------------------------------------


def test_golden_scanner_vcp_detected(monkeypatch, golden_update):
    """Scanner case: VCP detected via monkeypatched legacy detector.

    350-bar uptrend with mocked detect_vcp returning high-confidence
    VCP signal (score 82.5) — quality_score passthrough.
    """
    n = 350
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)
    close = np.linspace(80, 130, n)
    last_close = float(close[-1])

    price_df = golden_ohlcv_frame(index=index, close=close)
    bench_df = golden_benchmark_frame(index=index)
    data = golden_stock_data(
        symbol="GOLDEN_SCAN_VCP",
        price_data=price_df,
        benchmark_data=bench_df,
    )

    def mock_detect_vcp(self, prices, volumes):
        return {
            "vcp_detected": True,
            "vcp_score": 82.5,
            "num_bases": 4,
            "contracting_depth": True,
            "contraction_ratio": 0.62,
            "depth_score": 88.0,
            "contracting_volume": True,
            "volume_score": 74.0,
            "tight_near_highs": True,
            "tightness_score": 90.0,
            "atr_score": 71.0,
            "atr_contraction_ratio": 0.68,
            "pivot_info": {
                "pivot": 132.25,
                "distance_pct": 1.8,
                "ready_for_breakout": True,
            },
            "current_price": last_close,
            "distance_from_high_pct": 1.2,
        }

    from app.analysis.patterns import vcp_wrapper

    monkeypatch.setattr(vcp_wrapper.VCPDetector, "detect_vcp", mock_detect_vcp)

    result = _SCANNER.scan_stock("GOLDEN_SCAN_VCP", data)
    maybe_export_snapshot("scanner_vcp_detected", result, golden_update, is_scanner=True)

    assert_golden_scanner_match(result, {
        "screener_name": "setup_engine",
        # Actual: score=72.4, quality=75, readiness=68.6
        "score_range": (61, 84),
        "rating": "Watch",
        "pattern_primary": "vcp",
        "pivot_type": "vcp_pivot",
        "setup_score": (61, 84),
        "quality_score": (64, 87),
        "readiness_score": (58, 79),
        "required_readiness_fields": _STANDARD_READINESS_FIELDS,
    })


# ---------------------------------------------------------------------------
# Edge / negative cases (5)
# ---------------------------------------------------------------------------


def _build_scanner_no_pattern_exponential() -> tuple[dict[str, Any], dict[str, Any]]:
    """Edge case: No pattern detected — exponential growth defeats all detectors.

    350 bars of constant 0.6%/day growth. The uniform percentage change
    produces no consolidation, pullback, or tight zone for any detector.
    """
    n = 350
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)

    close = 50.0 * (1.006 ** np.arange(n))

    price_df = golden_ohlcv_frame(index=index, close=close)
    bench_df = golden_benchmark_frame(index=index)
    data = golden_stock_data(
        symbol="GOLDEN_SCAN_NOPATTERN",
        price_data=price_df,
        benchmark_data=bench_df,
    )

    fixture = {"symbol": "GOLDEN_SCAN_NOPATTERN", "data": data}
    expectation = {
        "score_range": (0, 0),
        "passes": False,
        "rating": "Pass",
        "screener_name": "setup_engine",
        "pattern_primary_is_none": True,
        "failed_contains": ["no_primary_pattern"],
    }
    return fixture, expectation


def _build_scanner_insufficient_50_bars() -> tuple[dict[str, Any], dict[str, Any]]:
    """Edge case: Early guard — only 50 daily bars (need ≥100).

    The scanner's early guard rejects before building any payload.
    """
    n = 50
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)
    close = np.linspace(100, 120, n)

    price_df = golden_ohlcv_frame(index=index, close=close)
    data = golden_stock_data(
        symbol="GOLDEN_SCAN_SHORT",
        price_data=price_df,
    )

    fixture = {"symbol": "GOLDEN_SCAN_SHORT", "data": data}
    expectation = {
        "rating": "Insufficient Data",
        "score_range": (0, 0),
        "passes": False,
        "screener_name": "setup_engine",
        "has_setup_engine": False,
    }
    return fixture, expectation


def _build_scanner_policy_insufficient() -> tuple[dict[str, Any], dict[str, Any]]:
    """Edge case: Policy failure — 200 bars passes 100-bar guard but fails 252-bar policy.

    The scanner builds a payload with all scores null but does not crash.
    """
    n = 200
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)
    close = np.linspace(80, 110, n)

    price_df = golden_ohlcv_frame(index=index, close=close)
    bench_df = golden_benchmark_frame(index=index)
    data = golden_stock_data(
        symbol="GOLDEN_SCAN_POLICY",
        price_data=price_df,
        benchmark_data=bench_df,
    )

    fixture = {"symbol": "GOLDEN_SCAN_POLICY", "data": data}
    expectation = {
        "rating": "Insufficient Data",
        "score_range": (0, 0),
        "passes": False,
        "screener_name": "setup_engine",
        # Payload exists but all scores are null
        "pattern_primary_is_none": True,
        "nullable_readiness_fields": [
            "setup_score", "quality_score", "readiness_score",
        ],
    }
    return fixture, expectation


def _build_scanner_high_score_not_ready() -> tuple[dict[str, Any], dict[str, Any]]:
    """Edge case: Gate failure despite quality — close spiked 10% above pivot.

    350 bars with a 3WT tight zone at bars 300-349 (recent, high quality),
    but the last bar's close is spiked 10% above the tight zone.
    This makes distance_to_pivot_pct ≈ +10%, far outside [-2%, 3%].
    Multiple gates may fail; we assert ``outside_early_zone`` is present.
    """
    rng = np.random.default_rng(42)
    n = 350
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)

    # Uptrend: 50 → 95 over 300 bars
    uptrend = np.linspace(50, 95, 300)
    # Tight zone: 49 bars near 100 with ±0.3% jitter
    tight = 100.0 + rng.uniform(-0.3, 0.3, 49)
    # Last bar: spike 10% above tight zone
    spike = np.array([110.0])
    close = np.concatenate([uptrend, tight, spike])

    price_df = golden_ohlcv_frame(index=index, close=close)
    bench_df = golden_benchmark_frame(index=index)
    data = golden_stock_data(
        symbol="GOLDEN_SCAN_NOT_READY",
        price_data=price_df,
        benchmark_data=bench_df,
    )

    fixture = {"symbol": "GOLDEN_SCAN_NOT_READY", "data": data}
    expectation = {
        "passes": False,
        "screener_name": "setup_engine",
        "setup_ready": False,
        "failed_contains": ["outside_early_zone"],
    }
    return fixture, expectation


def _build_scanner_degraded_no_benchmark() -> tuple[dict[str, Any], dict[str, Any]]:
    """Edge case: Missing benchmark data — RS fields become null.

    Same 3WT data but with empty benchmark DataFrame. RS-based readiness
    fields are null, ``data_policy:degraded`` in invalidation_flags,
    but RS gate passes (permissive — both None passes).
    """
    rng = np.random.default_rng(42)
    n = 350
    end_date = pd.Timestamp("2023-06-16")
    index = pd.bdate_range(end=end_date, periods=n)

    uptrend = np.linspace(50, 95, 300)
    tight = 100.0 + rng.uniform(-0.3, 0.3, 50)
    close = np.concatenate([uptrend, tight])

    price_df = golden_ohlcv_frame(index=index, close=close)
    data = golden_stock_data(
        symbol="GOLDEN_SCAN_NO_BENCH",
        price_data=price_df,
        benchmark_data=pd.DataFrame(),
    )

    fixture = {"symbol": "GOLDEN_SCAN_NO_BENCH", "data": data}
    expectation = {
        "screener_name": "setup_engine",
        "invalidation_flags_contains": ["data_policy:degraded"],
        # RS gate is permissive: None → passes
        "passed_contains": ["rs_leadership_ok"],
        "nullable_readiness_fields": ["rs", "rs_vs_spy_65d"],
    }
    return fixture, expectation


# ---------------------------------------------------------------------------
# Parametrized edge cases (5)
# ---------------------------------------------------------------------------

_EDGE_CASES = [
    ("scanner_no_pattern_exponential", *_build_scanner_no_pattern_exponential()),
    ("scanner_insufficient_50_bars", *_build_scanner_insufficient_50_bars()),
    ("scanner_policy_insufficient", *_build_scanner_policy_insufficient()),
    ("scanner_high_score_not_ready", *_build_scanner_high_score_not_ready()),
    ("scanner_degraded_no_benchmark", *_build_scanner_degraded_no_benchmark()),
]


@pytest.mark.parametrize(
    "case_id, fixture, expectation",
    _EDGE_CASES,
    ids=[c[0] for c in _EDGE_CASES],
)
def test_golden_scanner_edge(case_id, fixture, expectation, golden_update):
    """Golden edge/negative scanner cases."""
    result = _SCANNER.scan_stock(
        fixture["symbol"],
        fixture["data"],
    )
    maybe_export_snapshot(case_id, result, golden_update, is_scanner=True)
    assert_golden_scanner_match(result, expectation)
