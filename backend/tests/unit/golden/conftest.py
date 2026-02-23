"""Shared fixtures and composable assertion helpers for golden regression tests.

Golden tests pin known-good detector outputs as inline expectations.
The ``--golden-update`` flag writes actual outputs to ``snapshots/`` for
review before manually updating inline expectations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from app.analysis.patterns.detectors import (
    DetectorOutcome,
    PatternDetectorInput,
    PatternDetectorResult,
)

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"

# Sentinel for "not specified" (distinct from None which is a valid value)
_SENTINEL = object()


# ---------------------------------------------------------------------------
# pytest CLI flag
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--golden-update",
        action="store_true",
        default=False,
        help="Export actual detector outputs to snapshots/ for review",
    )


@pytest.fixture
def golden_update(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--golden-update")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def golden_ohlcv_frame(
    *,
    index: pd.DatetimeIndex,
    close: np.ndarray | Sequence[float],
    low: np.ndarray | Sequence[float] | None = None,
    high: np.ndarray | Sequence[float] | None = None,
    volume: np.ndarray | Sequence[float] | None = None,
    extra_cols: dict[str, np.ndarray | Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame from close prices."""
    close_arr = np.asarray(close, dtype=float)
    n = len(close_arr)
    assert len(index) == n, f"index length {len(index)} != close length {n}"

    df = pd.DataFrame(
        {
            "Open": close_arr * 0.995,
            "High": close_arr * 1.01 if high is None else np.asarray(high, dtype=float),
            "Low": close_arr * 0.99 if low is None else np.asarray(low, dtype=float),
            "Close": close_arr,
            "Volume": (
                np.full(n, 1_000_000, dtype=float)
                if volume is None
                else np.asarray(volume, dtype=float)
            ),
        },
        index=index,
    )

    if extra_cols:
        for col_name, col_data in extra_cols.items():
            df[col_name] = np.asarray(col_data, dtype=float)

    return df


def golden_detector_input(
    *,
    symbol: str = "GOLDEN",
    timeframe: str = "daily",
    daily: pd.DataFrame | None = None,
    weekly: pd.DataFrame | None = None,
) -> PatternDetectorInput:
    """Wrap daily/weekly frames into a PatternDetectorInput."""
    features: dict[str, Any] = {}
    daily_bars = 0
    weekly_bars = 0

    if daily is not None:
        features["daily_ohlcv"] = daily
        daily_bars = len(daily)
    if weekly is not None:
        features["weekly_ohlcv"] = weekly
        weekly_bars = len(weekly)

    return PatternDetectorInput(
        symbol=symbol,
        timeframe=timeframe,
        daily_bars=daily_bars,
        weekly_bars=weekly_bars,
        features=features,
    )


# ---------------------------------------------------------------------------
# Composable assertion helpers
# ---------------------------------------------------------------------------


def assert_outcome(result: PatternDetectorResult, expected: str) -> None:
    """Exact match on result.outcome.value."""
    actual = result.outcome.value
    assert actual == expected, (
        f"outcome: expected {expected!r}, got {actual!r}"
    )


def assert_candidate_count(
    result: PatternDetectorResult,
    *,
    exact: int | None = None,
    min_count: int | None = None,
) -> None:
    """Assert candidate count (exact or minimum)."""
    actual = len(result.candidates)
    if exact is not None:
        assert actual == exact, (
            f"candidate_count: expected exactly {exact}, got {actual}"
        )
    if min_count is not None:
        assert actual >= min_count, (
            f"candidate_count: expected >= {min_count}, got {actual}"
        )


def assert_primary_fields(
    candidate: Mapping[str, Any],
    *,
    pattern: str | None = None,
    timeframe: str | None = None,
    pivot_type: str | None = None,
) -> None:
    """Exact string matches on structural candidate fields."""
    if pattern is not None:
        assert candidate["pattern"] == pattern, (
            f"pattern: expected {pattern!r}, got {candidate['pattern']!r}"
        )
    if timeframe is not None:
        assert candidate["timeframe"] == timeframe, (
            f"timeframe: expected {timeframe!r}, got {candidate['timeframe']!r}"
        )
    if pivot_type is not None:
        assert candidate["pivot_type"] == pivot_type, (
            f"pivot_type: expected {pivot_type!r}, got {candidate['pivot_type']!r}"
        )


def assert_pivot_type_contains(candidate: Mapping[str, Any], substring: str) -> None:
    """Assert pivot_type contains a substring."""
    actual = candidate.get("pivot_type") or ""
    assert substring in actual, (
        f"pivot_type {actual!r} does not contain {substring!r}"
    )


def assert_score_ranges(
    candidate: Mapping[str, Any],
    *,
    confidence: tuple[float, float] | None = None,
    quality_score: tuple[float, float] | None = None,
    readiness_score: tuple[float, float] | None = None,
) -> None:
    """Assert scores fall within (min, max) ranges."""
    if confidence is not None:
        val = candidate.get("confidence")
        assert val is not None, "confidence is None"
        lo, hi = confidence
        assert lo <= val <= hi, (
            f"confidence: expected [{lo}, {hi}], got {val}"
        )
    if quality_score is not None:
        val = candidate.get("quality_score")
        assert val is not None, "quality_score is None"
        lo, hi = quality_score
        assert lo <= val <= hi, (
            f"quality_score: expected [{lo}, {hi}], got {val}"
        )
    if readiness_score is not None:
        val = candidate.get("readiness_score")
        assert val is not None, "readiness_score is None"
        lo, hi = readiness_score
        assert lo <= val <= hi, (
            f"readiness_score: expected [{lo}, {hi}], got {val}"
        )


def assert_pivot_approx(
    candidate: Mapping[str, Any],
    *,
    approx: float,
    tolerance_pct: float,
) -> None:
    """Assert pivot_price is within tolerance_pct of approx."""
    val = candidate.get("pivot_price")
    assert val is not None, "pivot_price is None"
    diff_pct = abs(val - approx) / approx * 100.0
    assert diff_pct <= tolerance_pct, (
        f"pivot_price: expected ~{approx} +/-{tolerance_pct}%, got {val} "
        f"(diff={diff_pct:.2f}%)"
    )


def assert_checks(
    candidate: Mapping[str, Any],
    *,
    required_true: Sequence[str] | None = None,
    required_false: Sequence[str] | None = None,
) -> None:
    """Assert boolean checks on candidate['checks']."""
    checks = candidate.get("checks", {})
    if required_true:
        for key in required_true:
            assert checks.get(key) is True, (
                f"checks[{key!r}]: expected True, got {checks.get(key)!r}"
            )
    if required_false:
        for key in required_false:
            assert checks.get(key) is False, (
                f"checks[{key!r}]: expected False, got {checks.get(key)!r}"
            )


def assert_metrics_range(
    candidate: Mapping[str, Any],
    ranges_dict: dict[str, tuple[float, float]],
) -> None:
    """Assert metric values fall within (min, max) ranges."""
    metrics = candidate.get("metrics", {})
    for key, (lo, hi) in ranges_dict.items():
        val = metrics.get(key)
        assert val is not None, f"metrics[{key!r}] is None or missing"
        assert lo <= val <= hi, (
            f"metrics[{key!r}]: expected [{lo}, {hi}], got {val}"
        )


def assert_result_checks(
    result: PatternDetectorResult,
    *,
    passed_superset: Sequence[str] | None = None,
    failed_contains: Sequence[str] | None = None,
) -> None:
    """Assert result-level passed/failed checks."""
    if passed_superset:
        passed = set(result.passed_checks)
        for check in passed_superset:
            assert check in passed, (
                f"passed_checks missing {check!r}; have: {sorted(passed)}"
            )
    if failed_contains:
        failed = set(result.failed_checks)
        for check in failed_contains:
            assert check in failed, (
                f"failed_checks missing {check!r}; have: {sorted(failed)}"
            )


# ---------------------------------------------------------------------------
# Top-level orchestrators
# ---------------------------------------------------------------------------


def assert_golden_match(
    result: PatternDetectorResult,
    expectation: dict[str, Any],
) -> None:
    """Delegate to sub-functions based on expectation dict keys."""
    if "outcome" in expectation:
        assert_outcome(result, expectation["outcome"])

    if "candidate_count" in expectation:
        assert_candidate_count(result, exact=expectation["candidate_count"])
    if "candidate_count_min" in expectation:
        assert_candidate_count(result, min_count=expectation["candidate_count_min"])

    if "passed_checks_superset" in expectation:
        assert_result_checks(
            result, passed_superset=expectation["passed_checks_superset"]
        )
    if "failed_checks_contains" in expectation:
        assert_result_checks(
            result, failed_contains=expectation["failed_checks_contains"]
        )

    # Candidate-level assertions operate on the first candidate
    _CANDIDATE_KEYS = {
        "pattern", "pivot_type", "timeframe", "pivot_type_contains",
        "confidence", "quality_score", "readiness_score",
        "checks_true", "checks_false", "metrics_range", "pivot_approx",
    }
    candidate = result.candidates[0] if result.candidates else None

    if candidate is None and _CANDIDATE_KEYS & expectation.keys():
        raise AssertionError(
            "Expectation includes candidate-level keys "
            f"{sorted(_CANDIDATE_KEYS & expectation.keys())} "
            "but result has no candidates"
        )

    if candidate is not None:
        if "pattern" in expectation or "pivot_type" in expectation or "timeframe" in expectation:
            assert_primary_fields(
                candidate,
                pattern=expectation.get("pattern"),
                timeframe=expectation.get("timeframe"),
                pivot_type=expectation.get("pivot_type"),
            )
        if "pivot_type_contains" in expectation:
            assert_pivot_type_contains(candidate, expectation["pivot_type_contains"])
        if "confidence" in expectation:
            assert_score_ranges(candidate, confidence=expectation["confidence"])
        if "quality_score" in expectation:
            assert_score_ranges(candidate, quality_score=expectation["quality_score"])
        if "readiness_score" in expectation:
            assert_score_ranges(candidate, readiness_score=expectation["readiness_score"])
        if "checks_true" in expectation:
            assert_checks(candidate, required_true=expectation["checks_true"])
        if "checks_false" in expectation:
            assert_checks(candidate, required_false=expectation["checks_false"])
        if "metrics_range" in expectation:
            assert_metrics_range(candidate, expectation["metrics_range"])
        if "pivot_approx" in expectation:
            assert_pivot_approx(
                candidate,
                approx=expectation["pivot_approx"]["value"],
                tolerance_pct=expectation["pivot_approx"]["tolerance_pct"],
            )


# ---------------------------------------------------------------------------
# Aggregator-specific assertion helper
# ---------------------------------------------------------------------------


def assert_golden_aggregation_match(
    output: Any,  # AggregatedPatternOutput
    expectation: dict[str, Any],
) -> None:
    """Handle aggregator-level output assertions."""
    if "pattern_primary" in expectation:
        assert output.pattern_primary == expectation["pattern_primary"], (
            f"pattern_primary: expected {expectation['pattern_primary']!r}, "
            f"got {output.pattern_primary!r}"
        )

    if "pattern_primary_is_none" in expectation and expectation["pattern_primary_is_none"]:
        assert output.pattern_primary is None, (
            f"pattern_primary: expected None, got {output.pattern_primary!r}"
        )

    if "pattern_confidence" in expectation:
        lo, hi = expectation["pattern_confidence"]
        val = output.pattern_confidence
        if val is None:
            assert lo <= 0 and hi >= 0, (
                f"pattern_confidence is None, expected [{lo}, {hi}]"
            )
        else:
            assert lo <= val <= hi, (
                f"pattern_confidence: expected [{lo}, {hi}], got {val}"
            )

    if "candidate_count" in expectation:
        actual = len(output.candidates)
        assert actual == expectation["candidate_count"], (
            f"candidate_count: expected {expectation['candidate_count']}, got {actual}"
        )
    if "candidate_count_min" in expectation:
        actual = len(output.candidates)
        assert actual >= expectation["candidate_count_min"], (
            f"candidate_count: expected >= {expectation['candidate_count_min']}, "
            f"got {actual}"
        )

    if "passed_checks_superset" in expectation:
        passed = set(output.passed_checks)
        for check in expectation["passed_checks_superset"]:
            assert check in passed, (
                f"passed_checks missing {check!r}; have: {sorted(passed)}"
            )

    if "failed_checks_contains" in expectation:
        failed = set(output.failed_checks)
        for check in expectation["failed_checks_contains"]:
            assert check in failed, (
                f"failed_checks missing {check!r}; have: {sorted(failed)}"
            )

    if "detector_trace_count" in expectation:
        actual = len(output.detector_traces)
        assert actual == expectation["detector_trace_count"], (
            f"detector_trace_count: expected {expectation['detector_trace_count']}, "
            f"got {actual}"
        )


# ---------------------------------------------------------------------------
# Golden snapshot export
# ---------------------------------------------------------------------------


def _serialize_result(result: PatternDetectorResult) -> dict[str, Any]:
    """Convert detector result to JSON-serializable dict."""
    return {
        "outcome": result.outcome.value,
        "detector_name": result.detector_name,
        "candidate_count": len(result.candidates),
        "passed_checks": list(result.passed_checks),
        "failed_checks": list(result.failed_checks),
        "warnings": list(result.warnings),
        "candidates": [dict(c) for c in result.candidates],
    }


def _serialize_aggregation(output: Any) -> dict[str, Any]:
    """Convert aggregator output to JSON-serializable dict."""
    return {
        "pattern_primary": output.pattern_primary,
        "pattern_confidence": output.pattern_confidence,
        "pivot_price": output.pivot_price,
        "pivot_type": output.pivot_type,
        "candidate_count": len(output.candidates),
        "candidates": [dict(c) for c in output.candidates],
        "passed_checks": list(output.passed_checks),
        "failed_checks": list(output.failed_checks),
        "detector_trace_count": len(output.detector_traces),
        "detector_traces": [
            {
                "detector_name": t.detector_name,
                "outcome": t.outcome,
                "candidate_count": t.candidate_count,
            }
            for t in output.detector_traces
        ],
    }


def maybe_export_snapshot(
    case_id: str,
    result: Any,
    golden_update: bool,
    *,
    is_aggregation: bool = False,
    is_scanner: bool = False,
) -> None:
    """If --golden-update, write snapshot and skip the test."""
    if not golden_update:
        return

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS_DIR / f"{case_id}.json"

    if is_scanner:
        payload = _serialize_scanner_result(result)
    elif is_aggregation:
        payload = _serialize_aggregation(result)
    else:
        payload = _serialize_result(result)

    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    pytest.skip(f"Golden snapshot '{case_id}' written to {path}")


# ---------------------------------------------------------------------------
# Scanner-level fixture builders
# ---------------------------------------------------------------------------


def golden_stock_data(
    *,
    symbol: str,
    price_data: pd.DataFrame,
    benchmark_data: pd.DataFrame | None = None,
) -> Any:
    """Build a StockData instance with empty fundamentals (Setup Engine doesn't use them)."""
    from app.scanners.base_screener import StockData

    return StockData(
        symbol=symbol,
        price_data=price_data,
        benchmark_data=benchmark_data if benchmark_data is not None else pd.DataFrame(),
        fundamentals={},
        quarterly_growth={},
        earnings_history=None,
        fetch_errors={},
    )


def golden_benchmark_frame(
    *,
    index: pd.DatetimeIndex,
    start_price: float = 400.0,
    end_price: float = 530.0,
) -> pd.DataFrame:
    """Build SPY-like benchmark OHLCV on the same DatetimeIndex as price data."""
    close = np.linspace(start_price, end_price, len(index))
    return golden_ohlcv_frame(index=index, close=close)


def _expand_weekly_to_daily(
    weekly_df: pd.DataFrame,
    start_monday: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Expand weekly OHLCV into 5 daily bars per week.

    The expansion is designed so daily data resampled back to weekly
    reproduces the original weekly OHLCV values (Open, High, Low, Close, Volume).

    Args:
        weekly_df: Weekly OHLCV DataFrame (indexed by week-ending Friday).
        start_monday: Override the Monday of the first week. If None, derived
                      from the first weekly index date.

    Returns:
        Daily OHLCV DataFrame with business-day index.
    """
    rows: list[dict[str, Any]] = []

    for i, (week_date, row) in enumerate(weekly_df.iterrows()):
        if start_monday is not None and i == 0:
            monday = start_monday
        else:
            # Derive Monday from the week-ending date
            monday = pd.Timestamp(week_date) - pd.Timedelta(days=4)
        bdays = pd.bdate_range(monday, periods=5)

        w_open = float(row["Open"])
        w_close = float(row["Close"])
        w_high = float(row["High"])
        w_low = float(row["Low"])
        w_volume = float(row["Volume"])

        daily_vol = w_volume / 5.0

        for d in range(5):
            if d == 0:
                d_open = w_open
            else:
                d_open = rows[-1]["Close"]  # previous day's close

            if d == 4:
                d_close = w_close
            else:
                # Linear interpolation between open and close
                t = (d + 1) / 5.0
                d_close = w_open + t * (w_close - w_open)

            d_high = max(d_open, d_close) * 1.002
            d_low = min(d_open, d_close) * 0.998

            rows.append({
                "date": bdays[d],
                "Open": d_open,
                "High": d_high,
                "Low": d_low,
                "Close": d_close,
                "Volume": daily_vol,
            })

    # Overwrite high/low to ensure weekly resampling reproduces exact weekly H/L
    row_idx = 0
    for _, row in weekly_df.iterrows():
        w_high = float(row["High"])
        w_low = float(row["Low"])
        # Find day with max high in this week's 5 bars, set to weekly high
        max_h_idx = max(range(5), key=lambda d: rows[row_idx + d]["High"])
        rows[row_idx + max_h_idx]["High"] = w_high
        # Find day with min low, set to weekly low
        min_l_idx = min(range(5), key=lambda d: rows[row_idx + d]["Low"])
        rows[row_idx + min_l_idx]["Low"] = w_low
        row_idx += 5

    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df.pop("date"))
    return df


# ---------------------------------------------------------------------------
# Scanner-level composable assertion helpers
# ---------------------------------------------------------------------------


def assert_screener_result_fields(
    result: Any,
    *,
    rating: str | list[str] | None = None,
    passes: bool | None = None,
    score_range: tuple[float, float] | None = None,
    screener_name: str | None = None,
) -> None:
    """Top-level ScreenerResult field checks."""
    if rating is not None:
        acceptable = [rating] if isinstance(rating, str) else rating
        assert result.rating in acceptable, (
            f"rating: expected one of {acceptable!r}, got {result.rating!r}"
        )
    if passes is not None:
        assert result.passes == passes, (
            f"passes: expected {passes}, got {result.passes}"
        )
    if score_range is not None:
        lo, hi = score_range
        assert lo <= result.score <= hi, (
            f"score: expected [{lo}, {hi}], got {result.score}"
        )
    if screener_name is not None:
        assert result.screener_name == screener_name, (
            f"screener_name: expected {screener_name!r}, got {result.screener_name!r}"
        )


def assert_payload_pattern(
    payload: Mapping[str, Any],
    *,
    pattern_primary: str | None = _SENTINEL,
    pattern_primary_is_none: bool = False,
    pivot_type: str | None = _SENTINEL,
    pivot_type_contains: str | None = None,
    setup_ready: bool | None = None,
) -> None:
    """Assert setup_engine payload pattern fields."""
    if pattern_primary_is_none:
        assert payload["pattern_primary"] is None, (
            f"pattern_primary: expected None, got {payload['pattern_primary']!r}"
        )
    elif pattern_primary is not _SENTINEL:
        assert payload["pattern_primary"] == pattern_primary, (
            f"pattern_primary: expected {pattern_primary!r}, "
            f"got {payload['pattern_primary']!r}"
        )
    if pivot_type is not _SENTINEL:
        assert payload["pivot_type"] == pivot_type, (
            f"pivot_type: expected {pivot_type!r}, got {payload['pivot_type']!r}"
        )
    if pivot_type_contains is not None:
        actual = payload.get("pivot_type") or ""
        assert pivot_type_contains in actual, (
            f"pivot_type {actual!r} does not contain {pivot_type_contains!r}"
        )
    if setup_ready is not None:
        assert payload["setup_ready"] == setup_ready, (
            f"setup_ready: expected {setup_ready}, got {payload['setup_ready']}"
        )


def assert_payload_score_ranges(
    payload: Mapping[str, Any],
    *,
    setup_score: tuple[float, float] | None = None,
    quality_score: tuple[float, float] | None = None,
    readiness_score: tuple[float, float] | None = None,
) -> None:
    """Assert setup_engine payload score ranges."""
    for name, expected_range in [
        ("setup_score", setup_score),
        ("quality_score", quality_score),
        ("readiness_score", readiness_score),
    ]:
        if expected_range is not None:
            val = payload.get(name)
            assert val is not None, f"{name} is None"
            lo, hi = expected_range
            assert lo <= val <= hi, (
                f"{name}: expected [{lo}, {hi}], got {val}"
            )


def assert_payload_readiness_present(
    payload: Mapping[str, Any],
    *,
    required_fields: Sequence[str] | None = None,
    nullable_fields: Sequence[str] | None = None,
) -> None:
    """Check that readiness fields are present (non-null for required, exist for nullable)."""
    if required_fields:
        for field_name in required_fields:
            assert payload.get(field_name) is not None, (
                f"readiness field {field_name!r} is None (expected non-null)"
            )
    if nullable_fields:
        for field_name in nullable_fields:
            assert field_name in payload, (
                f"readiness field {field_name!r} missing from payload"
            )


def assert_explain_checks(
    payload: Mapping[str, Any],
    *,
    passed_contains: Sequence[str] | None = None,
    failed_contains: Sequence[str] | None = None,
    invalidation_flags_contains: Sequence[str] | None = None,
) -> None:
    """Assert gate check assertions on the explain sub-payload."""
    explain = payload.get("explain", {})
    if passed_contains:
        passed = set(explain.get("passed_checks", []))
        for check in passed_contains:
            assert check in passed, (
                f"explain.passed_checks missing {check!r}; have: {sorted(passed)}"
            )
    if failed_contains:
        failed = set(explain.get("failed_checks", []))
        for check in failed_contains:
            assert check in failed, (
                f"explain.failed_checks missing {check!r}; have: {sorted(failed)}"
            )
    if invalidation_flags_contains:
        raw_flags = explain.get("invalidation_flags", [])
        flags: set[str] = set()
        if isinstance(raw_flags, Sequence):
            for raw in raw_flags:
                if isinstance(raw, Mapping):
                    code = raw.get("code")
                    if code:
                        detail = raw.get("message")
                        flags.add(str(code))
                        if detail:
                            flags.add(f"{code}:{detail}")
                elif raw:
                    flags.add(str(raw))
        for flag in invalidation_flags_contains:
            assert flag in flags, (
                f"explain.invalidation_flags missing {flag!r}; have: {sorted(flags)}"
            )


def assert_golden_scanner_match(
    result: Any,
    expectation: dict[str, Any],
) -> None:
    """Top-level orchestrator for scanner-level golden assertions.

    Delegates to sub-helpers based on which keys are present in the
    expectation dict (same delegation pattern as ``assert_golden_match``).
    """
    # ScreenerResult-level assertions
    _RESULT_KEYS = {"rating", "passes", "score_range", "screener_name"}
    if _RESULT_KEYS & expectation.keys():
        assert_screener_result_fields(
            result,
            rating=expectation.get("rating"),
            passes=expectation.get("passes"),
            score_range=expectation.get("score_range"),
            screener_name=expectation.get("screener_name"),
        )

    # Payload-level assertions require setup_engine key
    payload = result.details.get("setup_engine") if hasattr(result, "details") else None

    if expectation.get("has_setup_engine") is False:
        assert "setup_engine" not in result.details, (
            "Expected no setup_engine key in details"
        )
        return

    if payload is None and any(
        k in expectation
        for k in (
            "pattern_primary", "pattern_primary_is_none", "pivot_type",
            "pivot_type_contains", "setup_ready", "setup_score",
            "quality_score", "readiness_score", "required_readiness_fields",
            "nullable_readiness_fields", "passed_contains", "failed_contains",
            "invalidation_flags_contains",
        )
    ):
        raise AssertionError(
            "Expectation includes payload keys but result.details has no 'setup_engine'"
        )

    if payload is not None:
        # Pattern assertions
        _PATTERN_KEYS = {
            "pattern_primary", "pattern_primary_is_none",
            "pivot_type", "pivot_type_contains", "setup_ready",
        }
        if _PATTERN_KEYS & expectation.keys():
            assert_payload_pattern(
                payload,
                pattern_primary=expectation.get("pattern_primary", _SENTINEL),
                pattern_primary_is_none=expectation.get("pattern_primary_is_none", False),
                pivot_type=expectation.get("pivot_type", _SENTINEL),
                pivot_type_contains=expectation.get("pivot_type_contains"),
                setup_ready=expectation.get("setup_ready"),
            )

        # Score range assertions
        _SCORE_KEYS = {"setup_score", "quality_score", "readiness_score"}
        if _SCORE_KEYS & expectation.keys():
            assert_payload_score_ranges(
                payload,
                setup_score=expectation.get("setup_score"),
                quality_score=expectation.get("quality_score"),
                readiness_score=expectation.get("readiness_score"),
            )

        # Readiness field assertions
        if "required_readiness_fields" in expectation or "nullable_readiness_fields" in expectation:
            assert_payload_readiness_present(
                payload,
                required_fields=expectation.get("required_readiness_fields"),
                nullable_fields=expectation.get("nullable_readiness_fields"),
            )

        # Explain check assertions
        _EXPLAIN_KEYS = {"passed_contains", "failed_contains", "invalidation_flags_contains"}
        if _EXPLAIN_KEYS & expectation.keys():
            assert_explain_checks(
                payload,
                passed_contains=expectation.get("passed_contains"),
                failed_contains=expectation.get("failed_contains"),
                invalidation_flags_contains=expectation.get("invalidation_flags_contains"),
            )


# ---------------------------------------------------------------------------
# Scanner-level snapshot serialization
# ---------------------------------------------------------------------------


def _serialize_scanner_result(result: Any) -> dict[str, Any]:
    """Convert ScreenerResult to JSON-serializable dict for scanner snapshots."""
    payload = result.details.get("setup_engine") if hasattr(result, "details") else None
    serialized: dict[str, Any] = {
        "score": result.score,
        "passes": result.passes,
        "rating": result.rating,
        "breakdown": dict(result.breakdown) if result.breakdown else {},
        "screener_name": result.screener_name,
    }
    if payload is not None:
        # Deep-copy payload with JSON-safe serialization
        safe_payload: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "candidates":
                safe_payload[key] = [dict(c) for c in value]
            elif key == "explain":
                safe_payload[key] = (
                    dict(value) if isinstance(value, Mapping) else value
                )
            else:
                safe_payload[key] = value
        serialized["setup_engine"] = safe_payload
    return serialized
