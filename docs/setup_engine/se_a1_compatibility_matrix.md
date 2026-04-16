# SE-A1 Compatibility Matrix and Integration Risk Audit

## Scope
This audit maps Setup Engine insertion points across scanner orchestration and query/repository layers.

## File-Level Compatibility Matrix
| Area | File | Current role | Setup Engine insertion seam | Compatibility risk |
| --- | --- | --- | --- | --- |
| Scanner orchestration | `backend/app/scanners/scan_orchestrator.py` | Runs registered screeners, combines result dict, persists as flat keys + `details`. | Add `details["setup_engine"]` payload in `_combine_results()`; keep existing top-level legacy keys unchanged. | Medium: accidental top-level key collisions if Setup Engine fields are flattened. |
| Screener registration | `backend/app/scanners/screener_registry.py` | Registers screener classes and resolves active list by name. | Optional future `setup_engine` screener registration if/when detector is promoted to first-class screener. | Low: non-breaking while Setup Engine remains metadata-only. |
| Scanner contract | `backend/app/scanners/base_screener.py` | Defines `ScreenerResult.details` and score/rating interface. | Keep Setup Engine output in `details` payload to avoid changing `ScreenerResult` signature. | Low: structure is already extensible via JSON details. |
| Bulk scan lifecycle | `backend/app/use_cases/scanning/run_bulk_scan.py` | Calls `StockScanner.scan_stock_multi()` and sends raw dict to repository mapper. | No interface change needed; Setup Engine contract must be dict-serializable and deterministic per symbol. | Medium: non-serializable values in Setup Engine can break chunk persistence. |
| Legacy scan persistence | `backend/app/infra/db/repositories/scan_result_repo.py` | Maps orchestrator dict to indexed columns + stores full `details` JSON blob. | Preserve Setup Engine under `details.setup_engine`; do not add fragile column extraction until query requirements stabilize. | Medium: unexpected numeric/string types can break normalization. |
| Legacy query map | `backend/app/infra/query/scan_result_query.py` | Supports indexed sort/filter + selected JSON paths in `_JSON_FIELD_MAP`. | Add filter-critical Setup Engine fields as JSON paths like `$.setup_engine.setup_score`. | High: JSON-path naming drift causes silent filter/sort mismatches. |
| Feature snapshot build | `backend/app/use_cases/feature_store/build_daily_snapshot.py` | Stores full orchestrator dict as `details` for feature store rows. | Setup Engine is naturally carried through if attached to orchestrator result. | Low: already pass-through JSON semantics. |
| Feature store model | `backend/app/infra/db/models/feature_store.py` | Stores `details_json` blob in `stock_feature_daily`. | Keep Setup Engine nested under `details_json.setup_engine`; no schema migration required. | Low: JSON payload growth may impact row size/index strategy later. |
| Feature store repository bridge | `backend/app/infra/db/repositories/feature_store_repo.py` | Maps feature rows back to scan-domain DTOs via JSON reads. | Use top-level `setup_engine` keys for bridge filters/exports; avoid deep nested metric trees for filter-critical fields. | Medium: bridge logic must mirror legacy naming exactly. |
| Feature store query map | `backend/app/infra/query/feature_store_query.py` | SQL/json_extract filter and sort mapping for `details_json`. | Add Setup Engine JSON paths to `_JSON_FIELD_MAP` with numeric casting for sorting/filtering. | High: numeric JSON extraction must be cast consistently or sorts/filters drift. |

## Runtime Assumptions
1. Orchestrator output remains a JSON-serializable dict (no datetime/numpy objects after normalization).
2. Setup Engine contract lives under `details.setup_engine` and is not flattened into legacy top-level metric namespace.
3. Filter-critical Setup Engine metrics are direct children of `setup_engine` (not nested more deeply than one level).
4. Date fields use `YYYY-MM-DD`; timeframe uses `daily` or `weekly`.
5. Missing detector outputs are represented as `null` (not sentinel strings like `"N/A"`).

## Risk List and Mitigations
| Risk | Impact | Mitigation |
| --- | --- | --- |
| Naming drift between scanner payload and query map keys | Filters/sorts silently stop working. | Canonical field list in `backend/app/analysis/patterns/models.py`; enforce snake_case validation in payload builder. |
| Non-numeric JSON values for numeric fields | Query sort/filter inconsistency and cast failures. | Coerce numeric fields to float-or-null in payload assembly. |
| Over-nesting `setup_engine` metrics | Query mappers become brittle and expensive. | Keep filter-critical keys top-level under `setup_engine` only. |
| Ambiguous bool semantics (`setup_ready`, `rs_line_new_high`) | UI/query disagreement on readiness. | Define deterministic bool rules in `backend/app/scanners/setup_engine_scanner.py`. |
| Candidate payload key inconsistency | Frontend and downstream ETL parsing failures. | Validate candidate keys are snake_case and enforce date/timeframe normalization. |

## Recommended Safe Rollout Order
1. Land canonical contract types and payload builder (`SE-A2`).
2. Attach `setup_engine` payload to orchestrator details without changing query behavior.
3. Add query-map fields in `scan_result_query.py` and `feature_store_query.py` for filter-critical keys.
4. Add optional indexed column extraction only for proven hot paths.
