# ASIA v2 Data-Availability Launch Gate — 2026-04-16

- Bead: `StockScreenClaude-asia.11.7`
- Scope: explicit launch-gate evidence for unsupported/computed field transparency, cadence-fallback policy, benchmark mappings, and calendar correctness before GA progression
- Verdict: **PASS**
- Status: **evidence recorded and gate satisfied**

## Gate intent

This gate operationalizes the remaining non-price launch risks that are easy to hide in a cross-market rollout:

- unsupported or computed fields becoming opaque to API/UI consumers
- cadence fallback behavior drifting away from policy
- benchmark or calendar mappings silently regressing by market

Per `asia_v2_launch_gate_charter.md`, GA progression must stop if unsupported-field behavior becomes opaque, cadence fallback violates policy, or benchmark/calendar mappings are unverified.

## Evidence executed

The following focused verification suite was executed on 2026-04-16 under the project virtualenv:

```bash
cd backend
export DATABASE_URL=postgresql://stockscanner:stockscanner@127.0.0.1:55435/stockscanner_dummy
venv/bin/pytest \
  tests/unit/test_scan_field_availability.py \
  tests/unit/test_field_coverage_telemetry.py \
  tests/unit/test_benchmark_registry_service.py \
  tests/unit/test_market_calendar_service.py \
  tests/unit/test_market_calendar_service_engine.py \
  tests/parity/test_market_parity_e6.py -q
```

Result: **79 passed, 0 failed, 0 skipped**.

## Coverage by launch-gate concern

### 1. Unsupported/computed field transparency

Evidence source:
- `backend/tests/unit/test_scan_field_availability.py`

What it verifies:
- `field_availability` merges ownership/sentiment + cadence-fallback signals into one response surface
- unsupported fields on HK/JP/TW emit explicit `status` and `reason_code`
- computed growth fallback (`comparable_period_yoy`) is surfaced as `computed`, not silently treated as supported
- feature-store and legacy scan paths emit the same transparency metadata
- HTTP schema exposes `field_availability`, `growth_reporting_cadence`, and `growth_metric_basis`

Gate outcome:
- **PASS** — unsupported/computed behavior is explicit in the API surface and covered end-to-end through scan row mapping.

### 2. Cadence-fallback policy and unsupported-field telemetry

Evidence source:
- `backend/tests/unit/test_field_coverage_telemetry.py`
- `docs/asia/asia_v2_operator_runbooks.md` (RB-06)
- `docs/asia/asia_v2_governance_report.md`

What it verifies:
- `unsupported_field_ratio(...)` and `cadence_fallback_ratio(...)` compute correctly, including zero-universe handling
- per-market field-coverage snapshots are emitted from the registry with correct support-state counts and unsupported-field names
- alert thresholds classify `FIELD_COVERAGE` regressions by market
- weekly audit persists `worst_unsupported_ratio` and `worst_cadence_fallback_ratio`
- operator runbook RB-06 defines diagnosis, rollback, and recovery for field-coverage incidents

Gate outcome:
- **PASS** — cadence fallback and unsupported-field regressions are observable, thresholded, and operationally documented.

### 3. Benchmark mapping and calendar correctness

Evidence source:
- `backend/tests/unit/test_benchmark_registry_service.py`
- `backend/tests/unit/test_market_calendar_service.py`
- `backend/tests/unit/test_market_calendar_service_engine.py`
- `backend/tests/parity/test_market_parity_e6.py`

What it verifies:
- benchmark registry canonical mappings remain US→SPY, HK→^HSI, JP→^N225, TW→^TWII
- market calendar identifiers remain XNYS/XHKG/XTKS/XTAI
- weekend and session boundary behavior stays correct across supported markets
- parity suite reasserts benchmark normalization and candidate symbol mappings

Defect found while executing the gate:
- `MarketCalendarService.last_completed_trading_day()` incorrectly called `previous_session()` with a non-session Sunday date under `exchange_calendars`, causing the engine-level Sunday regression test to fail.

Resolution:
- fixed `backend/app/services/market_calendar_service.py` to use `date_to_session(..., direction="previous")` on non-session dates
- reran the full focused gate suite to green

Gate outcome:
- **PASS** — benchmark and calendar mappings are now verified and green under the current branch state.

## Exit criteria assessment

| Requirement | Evidence | Result |
|---|---|---|
| Unsupported-field behavior correctly surfaced | `test_scan_field_availability.py` | PASS |
| Cadence fallback policy observable and thresholded | `test_field_coverage_telemetry.py`, RB-06, weekly audit contract | PASS |
| Benchmark mappings verified | `test_benchmark_registry_service.py`, `test_market_parity_e6.py` | PASS |
| Calendar correctness verified | `test_market_calendar_service.py`, `test_market_calendar_service_engine.py` | PASS |

## Decision

> **PASS** — The data-availability launch gate is satisfied. Unsupported/computed field behavior is explicit, cadence fallback is observable and policy-backed, and benchmark/calendar mappings are verified under the current branch state.

This record closes `StockScreenClaude-asia.11.7` and removes the last formal blocker to `StockScreenClaude-asia.11.6` GA signoff.
