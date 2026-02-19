# Setup Engine Canonical Contract (v1)

This document is the canonical `setup_engine` contract for persistence and frontend consumption.

## Naming Policy
- Use `snake_case` for all top-level and nested keys.
- Any key used by query maps/UI filters must be a direct child of `setup_engine`.
- Date format is `YYYY-MM-DD`.
- Timeframe enum is `daily` or `weekly`.

## Bool Semantics
- `setup_ready`:
  - `true` when `readiness_score >= readiness_score_ready_min_pct` from parameter config and `failed_checks` is empty.
  - `false` otherwise.
- `rs_line_new_high`:
  - `true` when RS line makes a new high in the selected timeframe window.

## Nullability Rules
- Numeric fields are `null` when the detector cannot compute a stable value.
- `candidates` is always present (empty list when unavailable).
- `explain` is always present with all required child keys.

## Top-Level Schema
| Field | Type | Nullable | Unit | Source module |
| --- | --- | --- | --- | --- |
| `schema_version` | `str` | no | - | `backend/app/analysis/patterns/models.py` |
| `timeframe` | `daily \| weekly` | no | - | `backend/app/scanners/setup_engine_scanner.py` |
| `setup_score` | `float` | yes | `pct` | `backend/app/scanners/setup_engine_scanner.py` |
| `quality_score` | `float` | yes | `pct` | `backend/app/scanners/setup_engine_scanner.py` |
| `readiness_score` | `float` | yes | `pct` | `backend/app/scanners/setup_engine_scanner.py` |
| `setup_ready` | `bool` | no | - | `backend/app/scanners/setup_engine_scanner.py` |
| `pattern_primary` | `str` | yes | - | `backend/app/scanners/setup_engine_scanner.py` |
| `pattern_confidence` | `float` | yes | `pct` | `backend/app/scanners/setup_engine_scanner.py` |
| `pivot_price` | `float` | yes | `price` | `backend/app/scanners/setup_engine_scanner.py` |
| `pivot_type` | `str` | yes | - | `backend/app/scanners/setup_engine_scanner.py` |
| `pivot_date` | `YYYY-MM-DD` | yes | - | `backend/app/scanners/setup_engine_scanner.py` |
| `distance_to_pivot_pct` | `float` | yes | `pct` | `backend/app/scanners/setup_engine_scanner.py` |
| `atr14_pct` | `float` | yes | `pct` | `backend/app/scanners/setup_engine_scanner.py` |
| `bb_width_pctile_252` | `float` | yes | `pct` | `backend/app/scanners/setup_engine_scanner.py` |
| `volume_vs_50d` | `float` | yes | `ratio` | `backend/app/scanners/setup_engine_scanner.py` |
| `rs_line_new_high` | `bool` | no | - | `backend/app/scanners/setup_engine_scanner.py` |
| `candidates` | `list[PatternCandidate]` | no | - | `backend/app/scanners/setup_engine_scanner.py` |
| `explain` | `SetupEngineExplain` | no | - | `backend/app/scanners/setup_engine_scanner.py` |

## `candidates[]` Schema
| Field | Type | Nullable | Unit |
| --- | --- | --- | --- |
| `pattern` | `str` | no | - |
| `confidence_pct` | `float` | yes | `pct` |
| `pivot_price` | `float` | yes | `price` |
| `pivot_type` | `str` | yes | - |
| `pivot_date` | `YYYY-MM-DD` | yes | - |
| `distance_to_pivot_pct` | `float` | yes | `pct` |
| `setup_score` | `float` | yes | `pct` |
| `quality_score` | `float` | yes | `pct` |
| `readiness_score` | `float` | yes | `pct` |
| `timeframe` | `daily \| weekly` | no | - |

## `explain` Schema
| Field | Type | Nullable | Unit |
| --- | --- | --- | --- |
| `passed_checks` | `list[str]` | no | - |
| `failed_checks` | `list[str]` | no | - |
| `key_levels` | `dict[str, float \| null]` | no | `price` |
| `invalidation_flags` | `list[str]` | no | - |

## Canonical Placement
- Contract types and validators: `backend/app/analysis/patterns/models.py`
- Parameter governance and threshold bounds: `backend/app/analysis/patterns/config.py`
- Data requirements and incomplete-data policy: `backend/app/analysis/patterns/policy.py`
- Payload assembly and semantics: `backend/app/scanners/setup_engine_scanner.py`
