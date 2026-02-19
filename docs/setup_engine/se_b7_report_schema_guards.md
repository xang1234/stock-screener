# SE-B7 SetupEngine Report Schema Types and Serialization Guards

## Canonical Module
- `backend/app/analysis/patterns/report.py`

## Typed Schemas
- `SetupEngineReport` (top-level report container)
- `ExplainPayload`
- `KeyLevels`
- `InvalidationFlag`

## Boundary Guards
- Producer boundary:
  - `backend/app/scanners/setup_engine_scanner.py` now validates final payload with `assert_valid_setup_engine_report_payload(...)`.
  - Added `build_setup_engine_payload_from_report(...)` for typed report serialization.
- Persistence boundary:
  - `backend/app/infra/db/repositories/scan_result_repo.py` validates incoming `setup_engine` and drops invalid payloads with explicit validation errors attached.
- Query consumer boundary:
  - Payload strictness guarantees JSON-primitive serialization and unit/type consistency for downstream query/front-end readers.

## Validation Rules
- Required payload keys and naming policy (`snake_case`) enforced.
- Numeric unit-bearing fields must be numeric or null.
- Candidate confidence consistency enforced (`confidence` 0..1 and `confidence_pct` 0..100).
- Recursive JSON-serializable primitive check across payload.

## Canonical Example
- `canonical_setup_engine_report_examples()` provides regression fixtures used by tests.

## Validation Targets
- `backend/tests/unit/test_setup_engine_report_schema.py`
