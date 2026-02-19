"""DataFrame normalization and validation guards for detector inputs.

This module provides a single path for OHLCV preprocessing:
- chronological sorting
- required-column validation
- minimum-bar checks
- deterministic NaN handling

All functions are pure and return explicit check/warning/failure metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import pandas as pd


REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)


@dataclass(frozen=True)
class NormalizedOHLCV:
    """Normalized OHLCV frame plus prerequisite check metadata."""

    timeframe: Literal["daily", "weekly"]
    frame: pd.DataFrame | None
    checks: dict[str, bool]
    failed_checks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def prerequisites_ok(self) -> bool:
        return len(self.failed_checks) == 0


def normalize_ohlcv_frame(
    frame: pd.DataFrame,
    *,
    timeframe: Literal["daily", "weekly"],
    min_bars: int,
    required_columns: Sequence[str] = REQUIRED_OHLCV_COLUMNS,
) -> NormalizedOHLCV:
    """Normalize an OHLCV frame and return explicit prerequisite checks."""
    checks: dict[str, bool] = {
        "frame_present": True,
        "datetime_index": False,
        "required_columns": False,
        "min_bars": False,
        "nan_policy_applied": True,
    }
    failed: list[str] = []
    warnings: list[str] = []

    if not isinstance(frame, pd.DataFrame):
        return NormalizedOHLCV(
            timeframe=timeframe,
            frame=None,
            checks=checks,
            failed_checks=("invalid_ohlcv_frame_type",),
            warnings=(),
        )

    if frame.empty:
        return NormalizedOHLCV(
            timeframe=timeframe,
            frame=None,
            checks=checks,
            failed_checks=("empty_ohlcv_frame", f"{timeframe}_bars_lt_{min_bars}"),
            warnings=(),
        )

    df = frame.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        return NormalizedOHLCV(
            timeframe=timeframe,
            frame=None,
            checks=checks,
            failed_checks=("invalid_datetime_index",),
            warnings=(),
        )

    checks["datetime_index"] = True

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        warnings.append("ohlcv_sorted_chronologically")

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
        warnings.append("duplicate_timestamps_dropped")

    # Remove timezone to avoid hidden timezone assumptions across detectors.
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
        warnings.append("datetime_index_timezone_removed")

    lower_lookup = {str(col).lower(): str(col) for col in df.columns}
    missing_columns: list[str] = []
    rename_map: dict[str, str] = {}

    for canonical in required_columns:
        source = lower_lookup.get(canonical.lower())
        if source is None:
            missing_columns.append(canonical)
        elif source != canonical:
            rename_map[source] = canonical

    if missing_columns:
        checks["required_columns"] = False
        failed.extend(
            [f"missing_column_{col.lower()}" for col in missing_columns]
        )
        return NormalizedOHLCV(
            timeframe=timeframe,
            frame=None,
            checks=checks,
            failed_checks=tuple(failed),
            warnings=tuple(warnings),
        )

    if rename_map:
        df = df.rename(columns=rename_map)

    checks["required_columns"] = True
    required = list(required_columns)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    nan_rows = int(df[required].isna().any(axis=1).sum())
    if nan_rows > 0:
        df = df.dropna(subset=required)
        warnings.append(f"nan_rows_dropped:{nan_rows}")

    checks["min_bars"] = len(df) >= min_bars
    if not checks["min_bars"]:
        failed.append(f"{timeframe}_bars_lt_{min_bars}")

    return NormalizedOHLCV(
        timeframe=timeframe,
        frame=df if len(df) > 0 else None,
        checks=checks,
        failed_checks=tuple(failed),
        warnings=tuple(warnings),
    )


def normalize_detector_input_ohlcv(
    *,
    features: Mapping[str, Any],
    timeframe: Literal["daily", "weekly"],
    min_bars: int,
    feature_key: str,
    fallback_bar_count: int,
) -> NormalizedOHLCV:
    """Normalize detector input frame; fall back to bar counts when absent.

    This allows current stubs to stay compile-safe before scanner integration
    starts passing DataFrames through ``features``.
    """
    raw_frame = features.get(feature_key)
    if raw_frame is None:
        checks = {
            "frame_present": False,
            "datetime_index": False,
            "required_columns": False,
            "min_bars": fallback_bar_count >= min_bars,
            "nan_policy_applied": False,
        }
        failed = []
        if fallback_bar_count < min_bars:
            failed.append(f"{timeframe}_bars_lt_{min_bars}")
        warnings = ("missing_ohlcv_frame_using_bar_count_fallback",)
        return NormalizedOHLCV(
            timeframe=timeframe,
            frame=None,
            checks=checks,
            failed_checks=tuple(failed),
            warnings=warnings,
        )

    return normalize_ohlcv_frame(
        raw_frame,
        timeframe=timeframe,
        min_bars=min_bars,
    )
