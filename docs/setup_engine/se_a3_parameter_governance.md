# SE-A3 Parameter Governance (Setup Engine)

## Objectives
- Externalize setup thresholds from scanner logic.
- Enforce numeric bounds and contradiction guardrails.
- Preserve a documented rationale for strict/relaxed defaults.

## Canonical Source
- `backend/app/analysis/patterns/config.py`

## Strict vs Relaxed Defaults
| Parameter | Strict | Relaxed | Unit | Why |
| --- | --- | --- | --- | --- |
| `three_weeks_tight_max_contraction_pct_*` | `1.0` | `1.5` | `pct` | Strict mode preserves only very tight 3WT structures; relaxed mode handles normal leadership noise. |
| `squeeze_bb_width_pctile_max_*` | `20.0` | `35.0` | `pct` | Strict squeeze expects extreme compression; relaxed squeeze allows broader consolidation regimes. |

## Baseline Defaults
- `readiness_score_ready_min_pct = 70.0`
- `quality_score_min_pct = 60.0`
- `pattern_confidence_min_pct = 55.0`
- `early_zone_distance_to_pivot_pct_min = -2.0`
- `early_zone_distance_to_pivot_pct_max = 3.0`
- `atr14_pct_max_for_ready = 8.0`
- `volume_vs_50d_min_for_ready = 1.0`

## Guardrails
- Each parameter has bounded min/max ranges.
- Strict profile values must not exceed their relaxed counterparts.
- Early-zone min must be `<=` early-zone max.
- Quality threshold must not exceed readiness threshold.
- Unknown override keys are rejected.

## Runtime Usage
- Scanner payload assembly defaults to `DEFAULT_SETUP_ENGINE_PARAMETERS`.
- Runtime overrides must pass `build_setup_engine_parameters(overrides)` validation before use.
