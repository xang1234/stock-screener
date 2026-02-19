# SE-B1 Patterns Package Skeleton and Boundaries

## Package Layout
- `backend/app/analysis/patterns/__init__.py`
- `backend/app/analysis/patterns/models.py`
- `backend/app/analysis/patterns/config.py`
- `backend/app/analysis/patterns/policy.py`
- `backend/app/analysis/patterns/aggregator.py`
- `backend/app/analysis/patterns/detectors/base.py`
- `backend/app/analysis/patterns/detectors/cup_with_handle.py`
- `backend/app/analysis/patterns/detectors/vcp.py`
- `backend/app/analysis/patterns/detectors/double_bottom.py`
- `backend/app/analysis/patterns/detectors/__init__.py`

## Ownership Boundaries
- Analysis layer owns:
  - Pattern detector interfaces and implementations.
  - Aggregation logic and candidate arbitration.
  - Parameter validation and data sufficiency policy.
- Scanner layer owns:
  - Runtime integration with orchestrator and stock data fetch paths.
  - Final `setup_engine` payload assembly and persistence wiring.

## Dependency Rules
- `analysis/patterns/*` must not import from `app.scanners.*`.
- `detectors/*` import only analysis-layer modules (`models`, `config`, `policy`, detector base).
- `aggregator.py` consumes detector interfaces and emits normalized output; no DB/task/query dependencies.

## Circular Dependency Prevention
- `models.py` is foundational and does not depend on detectors or scanner modules.
- `config.py` and `policy.py` depend only on foundational models/constants.
- `detectors/*` depend on `config.py` and `detectors/base.py`.
- `aggregator.py` depends on detectors and policy, but nothing depends back on aggregator.

## Current Stub Scope
Detector modules are intentionally non-throwing stubs for interface stabilization. Concrete pattern math is deferred to SE-B6 and related B-series issues.
