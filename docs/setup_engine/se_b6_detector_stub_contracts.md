# SE-B6 Detector and Aggregator Stub Contracts

## Explicit Entry Modules
- `backend/app/analysis/patterns/vcp_wrapper.py` (`TODO(SE-C1)`)
- `backend/app/analysis/patterns/three_weeks_tight.py` (`TODO(SE-C2)`)
- `backend/app/analysis/patterns/high_tight_flag.py` (implemented through SE-C3b)
- `backend/app/analysis/patterns/cup_handle.py` (implemented through SE-C4b)
- `backend/app/analysis/patterns/nr7_inside_day.py` (`TODO(SE-C5)`)
- `backend/app/analysis/patterns/first_pullback.py` (implemented through SE-C6b)

## Interface Boundary
- Detector contract: `backend/app/analysis/patterns/detectors/base.py`
- Aggregator contract: `backend/app/analysis/patterns/aggregator.py`

All stubs are compile-safe and deterministic:
- return typed `PatternDetectorResult`
- emit explicit `insufficient_data` checks for no-data cases
- avoid scanner-layer imports

## Public API Surface
- Wildcard exports in `backend/app/analysis/patterns/__init__.py` are restricted to:
  - `PatternCandidate` schema helpers
  - shared technical utility functions
  - detector entrypoint classes / interfaces
