# SE-B4 DataFrame Normalization and Validation Guards

## Canonical Module
- `backend/app/analysis/patterns/normalization.py`

## Guard Responsibilities
- Enforce chronological index ordering.
- Enforce required OHLCV columns (`Open`, `High`, `Low`, `Close`, `Volume`) with case-insensitive normalization.
- Enforce minimum bar prerequisites per timeframe.
- Enforce deterministic NaN policy by dropping rows containing NaNs in required columns and emitting warning metadata.

## Key APIs
- `normalize_ohlcv_frame(...)`
  - Returns normalized frame plus `checks`, `failed_checks`, and `warnings`.
- `normalize_detector_input_ohlcv(...)`
  - Integrates feature-frame path with bar-count fallback for pre-integration stubs.

## Detector Behavior
Entry detectors now consume normalization results and return explicit non-detection when prerequisites fail:
- `failed_checks` include `insufficient_data` plus precise guard failure codes.
- `warnings` include normalization path details (`ohlcv_sorted_chronologically`, `nan_rows_dropped:*`, etc).

## Validation Targets
- `backend/tests/unit/test_pattern_normalization.py`
