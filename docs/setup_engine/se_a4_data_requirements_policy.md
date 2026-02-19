# SE-A4 Data Requirements and Incomplete-Data Policy

## Objectives
- Make Setup Engine behavior deterministic under missing/short data.
- Represent insufficiency/degradation explicitly in explain output.
- Prevent look-ahead bias from incomplete weekly bars.

## Canonical Source
- `backend/app/analysis/patterns/policy.py`

## Minimum Requirements
- `min_daily_bars = 252`
- `min_weekly_bars = 52`
- `min_benchmark_bars = 252` (for RS-dependent signals)
- `min_completed_sessions_in_current_week = 5`

## Deterministic Status Outcomes
- `ok`: all requirements satisfied.
- `degraded`: core pattern requirements pass, but benchmark history is missing/short or current week is incomplete.
- `insufficient`: daily/weekly minimum bars fail (or benchmark fails when degradation is not allowed).

## Incomplete Week Rule (No Look-Ahead)
- If current week sessions are below completeness threshold and incomplete weeks are not allowed:
  - `requires_weekly_exclude_current = true`
  - status is at least `degraded`
  - explain flags include `current_week_incomplete_exclude_from_weekly`

## Graceful Degradation Contract
- Policy outputs explicit `failed_reasons` and `degradation_reasons`.
- Payload assembly can consume policy output and:
  - append deterministic `failed_checks` / `invalidation_flags`
  - null out numeric setup fields when status is `insufficient`
- Detector exceptions should be converted to diagnostics (not uncaught runtime errors).
