# SE-H3: Maintainer Handoff and Follow-Up Backlog

> **Orientation guide** for new contributors inheriting the Setup Engine codebase. Part A covers architecture, design rationale, and extension guardrails so you can navigate the system without reverse-engineering it. Part B captures deferred v1.1 enhancements as a prioritized backlog.

## Table of Contents

**Part A — Maintainer Handoff**

1. [System Overview](#1-system-overview)
2. [New Contributor Quickstart](#2-new-contributor-quickstart)
3. [Architecture Map](#3-architecture-map)
4. [Design Philosophy and Key Decisions](#4-design-philosophy-and-key-decisions)
5. [Extension Guardrails](#5-extension-guardrails)
6. [Quality Gate Reference](#6-quality-gate-reference)

**Part B — Follow-Up Backlog**

7. [Backlog Overview](#7-backlog-overview)
8. [Enhancement Details](#8-enhancement-details)

**Appendices**

9. [Document Cross-Reference Index](#9-document-cross-reference-index)
10. [Keeping This Document Current](#10-keeping-this-document-current)

### Related Design Docs

- [SE-H1: Parameter Catalog and Calibration Report](se_h1_parameter_catalog.md) — Definitive reference for all 18 configurable thresholds, compile-time constants, and calibration profiles
- [SE-H2: Staged Rollout Plan](se_h2_staged_rollout_plan.md) — Operator playbook for safely enabling SE with feature flags and observability metrics
- [SE-H4: Formula Dictionary and Operator Runbook](se_h4_formula_dictionary_operator_runbook.md) — Field-level reference for all 31 output fields and diagnostic troubleshooting procedures
- [SE-A3: Parameter Governance](se_a3_parameter_governance.md) — Rationale for the parameter dataclass and validation architecture
- [SE-A4: Data Requirements Policy](se_a4_data_requirements_policy.md) — Incomplete-data behavior and graceful degradation semantics
- [SE-C7: Cross-Detector Calibration](se_c7_cross_detector_calibration.md) — Cross-detector normalization and confidence calibration
- [Setup Engine Contract v1](setup_engine_contract_v1.md) — Canonical schema contract for persistence and frontend consumption

### Canonical Source Files

| File | Role |
|------|------|
| `scanners/setup_engine_screener.py` | Main orchestrator — bridges StockData to analysis pipeline to ScreenerResult |
| `scanners/setup_engine_scanner.py` | Payload assembly + score synthesis via `build_setup_engine_payload()` |
| `analysis/patterns/config.py` | 18 runtime parameters, score weights, and validation guardrails |
| `analysis/patterns/models.py` | Domain types + 31 `SETUP_ENGINE_FIELD_SPECS` entries |
| `analysis/patterns/explain_builder.py` | 10-gate evaluation producing `ExplainResult` with `derived_ready` |

All paths relative to `backend/app/`.

---

## 1. System Overview

The Setup Engine (SE) is the 6th registered screener in the multi-screener stock scanning platform. Where the other five screeners (Minervini, CANSLIM, IPO, Volume Breakthrough, Custom) evaluate stocks against template criteria or fundamental metrics, SE discovers pre-breakout chart patterns and scores their readiness to trigger.

**Key dimensions:**

| Dimension | Count | Governed by |
|-----------|------:|-------------|
| Detectors | 7 | `detectors/__init__.py:default_pattern_detectors()` |
| Output fields | 31 | `models.py:SETUP_ENGINE_FIELD_SPECS` |
| Configurable parameters | 18 | `config.py:SetupEngineParameters` |
| Readiness gates | 10 | `explain_builder.py:build_explain_payload()` |
| Operational flags | 4 | `operational_flags.py:compute_operational_flags()` |

**Integration with scan orchestrator:** SE is registered via `@register_screener` on `SetupEngineScanner(BaseStockScreener)` in `scanners/setup_engine_screener.py:51`. It participates in composite scoring alongside the other five screeners. Adding SE changed the default equal-weight share from 1/5 to 1/6 per screener when all six are active. SE can be disabled entirely via the `settings.setup_engine_enabled` feature flag without affecting other screeners.

**Score synthesis formula:**

```
setup_score = 0.60 × quality_score + 0.40 × readiness_score
```

Source: `config.py:18` — `SETUP_SCORE_WEIGHTS = (0.60, 0.40)`

Per-candidate comparison uses a 3-way blend:

```
candidate_score = 0.55 × quality + 0.35 × readiness + 0.10 × confidence
```

Source: `config.py:16` — `CANDIDATE_SETUP_SCORE_WEIGHTS = (0.55, 0.35, 0.10)`

---

## 2. New Contributor Quickstart

### First 30 Minutes

| Step | Time | Action | Why |
|-----:|-----:|--------|-----|
| 1 | 5 min | Read [Setup Engine Contract v1](setup_engine_contract_v1.md) | Understand the persistence schema and field semantics before touching code |
| 2 | 5 min | Read [this document §4](#4-design-philosophy-and-key-decisions) | Absorb the five design principles so you don't fight the architecture |
| 3 | 2 min | Run `make gates` from the project root | Confirm all 5 quality gates pass on your machine |
| 4 | 10 min | Read [SE-H4 §8 Pipeline Walkthrough](se_h4_formula_dictionary_operator_runbook.md#8-pipeline-walkthrough) | Trace a single stock through the full SE pipeline end-to-end |
| 5 | 5 min | Read [SE-H1 §1 Quick-Reference Tuning Card](se_h1_parameter_catalog.md#1-quick-reference-tuning-card) | See all 18 parameters, their defaults, and allowed ranges at a glance |
| 6 | 3 min | Skim [this document §3](#3-architecture-map) | Map the 27 source files to their layers |

### Common Questions Lookup

| Question | Where to look |
|----------|---------------|
| What does field X mean? | [SE-H4 §1 Quick-Reference Field Index](se_h4_formula_dictionary_operator_runbook.md#1-quick-reference-field-index) |
| What is the default value of parameter Y? | [SE-H1 §1 Quick-Reference Tuning Card](se_h1_parameter_catalog.md#1-quick-reference-tuning-card) |
| How is `setup_score` calculated? | [SE-H4 §2 Score & Verdict Fields](se_h4_formula_dictionary_operator_runbook.md#2-score--verdict-fields) |
| Why does a stock show `setup_ready = false`? | [SE-H4 §6 Explain Payload Reference](se_h4_formula_dictionary_operator_runbook.md#6-explain-payload-reference) |
| What data does SE need? | [SE-A4 Data Requirements Policy](se_a4_data_requirements_policy.md) |
| How do I add a new detector? | [This document §5.1](#51-adding-a-detector) |
| How do I tune a threshold? | [SE-H1 §2 Score Synthesis Architecture](se_h1_parameter_catalog.md#2-score-synthesis-architecture) |
| How do I enable/disable SE? | [SE-H2 §2 Infrastructure Inventory](se_h2_staged_rollout_plan.md#2-infrastructure-inventory) |
| How do I regenerate golden snapshots? | [This document §6](#6-quality-gate-reference) — `make golden-update` |
| What are the operational flags? | [SE-H4 §6 Explain Payload Reference](se_h4_formula_dictionary_operator_runbook.md#6-explain-payload-reference) |
| How does SE integrate with the orchestrator? | [SE-A1 Compatibility Matrix](se_a1_compatibility_matrix.md) |
| What invariants must I preserve? | [This document §5.4](#54-invariants-table) |

---

## 3. Architecture Map

### File Inventory (27 files, ~7,700 lines)

Organized by layer from outermost (entry points) to innermost (infrastructure).

**Entry Points (2 files, 825 lines)**

| File | Lines | Role |
|------|------:|------|
| `scanners/setup_engine_screener.py` | 411 | Runtime entry point; bridges `StockData` → analysis pipeline → `ScreenerResult` |
| `scanners/setup_engine_scanner.py` | 414 | Payload assembly helpers; `build_setup_engine_payload()` score synthesis |

**Detectors (8 files, 2,658 lines)**

| File | Lines | Role |
|------|------:|------|
| `analysis/patterns/cup_handle.py` | 552 | Cup-with-handle detector |
| `analysis/patterns/high_tight_flag.py` | 455 | High-tight-flag detector |
| `analysis/patterns/first_pullback.py` | 447 | First-pullback / trend-resumption detector |
| `analysis/patterns/nr7_inside_day.py` | 325 | NR7 / inside-day trigger detector |
| `analysis/patterns/three_weeks_tight.py` | 317 | Three-weeks-tight / multi-weeks-tight detector |
| `analysis/patterns/vcp_wrapper.py` | 172 | VCP (volatility contraction pattern) detector wrapper |
| `analysis/patterns/detectors/double_bottom.py` | 71 | Double-bottom detector stub (`not_implemented`) |
| `analysis/patterns/detectors/base.py` | 319 | Detector interface, outcome taxonomy, graceful-failure semantics |

**Infrastructure + Registry (17 files, 4,238 lines)**

| File | Lines | Role |
|------|------:|------|
| `analysis/patterns/models.py` | 719 | Canonical schema: `SETUP_ENGINE_FIELD_SPECS` (31 entries) |
| `analysis/patterns/legacy_vcp_detection.py` | 519 | Legacy VCP detection (predecessor, retained for reference) |
| `analysis/patterns/technicals.py` | 376 | Shared technical utilities (ATR, Bollinger, RS, slope) |
| `analysis/patterns/aggregator.py` | 356 | Pattern aggregation; primary selection + tie-breaking |
| `analysis/patterns/calibration.py` | 327 | Cross-detector score normalization and confidence calibration |
| `analysis/patterns/report.py` | 319 | Typed report schemas and serialization guards |
| `analysis/patterns/config.py` | 315 | Parameter governance: 18 params, score weights, validation |
| `analysis/patterns/readiness.py` | 309 | Breakout-readiness feature computation (10 features) |
| `analysis/patterns/explain_builder.py` | 207 | 10-gate evaluation → `ExplainResult` |
| `analysis/patterns/normalization.py` | 195 | DataFrame normalization and validation guards |
| `analysis/patterns/trace.py` | 189 | Score trace builder for field-level auditability |
| `analysis/patterns/policy.py` | 138 | Data requirements and incomplete-data policy |
| `analysis/patterns/__init__.py` | 120 | Stable public APIs for the patterns package |
| `analysis/patterns/operational_flags.py` | 91 | 4 operational invalidation flags (informational) |
| `analysis/patterns/detectors/__init__.py` | 44 | Registry; `default_pattern_detectors()` defines execution order |
| `analysis/patterns/detectors/cup_with_handle.py` | 7 | Compatibility alias → `cup_handle.CupHandleDetector` |
| `analysis/patterns/detectors/vcp.py` | 7 | Compatibility alias → `vcp_wrapper.VCPWrapperDetector` |

### Simplified Data-Flow

```
StockData (price + benchmark)
    │
    ▼
SetupEngineScanner.scan_stock()          ← scanners/setup_engine_screener.py
    │
    ├─► Phase A: Data policy check       ← policy.py
    │       └─► Reject if insufficient
    │
    ├─► Phase B: Detector pipeline       ← aggregator.py
    │       ├─► 7 detectors (parallel-safe, run sequentially)
    │       │       └─► PatternCandidate[] per detector
    │       ├─► Calibration              ← calibration.py
    │       └─► Primary selection        ← aggregator.py (structural preference)
    │
    ├─► Phase B½: Context enrichment     ← setup_engine_screener.py
    │       └─► Stage, MA alignment, RS rating
    │
    └─► Phase C: Readiness + assembly    ← readiness.py, operational_flags.py
            ├─► 10 readiness features    ← readiness.py
            ├─► 4 operational flags      ← operational_flags.py
            ├─► 10-gate evaluation       ← explain_builder.py
            ├─► Score synthesis          ← setup_engine_scanner.py
            └─► SetupEnginePayload       ← models.py (31 fields)
```

Source: `scanners/setup_engine_screener.py:104-266`

Cross-ref: [SE-H4 §8 Pipeline Walkthrough](se_h4_formula_dictionary_operator_runbook.md#8-pipeline-walkthrough)

---

## 4. Design Philosophy and Key Decisions

Five principles guide every SE design choice. Understanding these prevents well-intentioned changes from violating the system's invariants.

### Design Principles

1. **Isolation** — SE is a self-contained screener. It does not read from or write to other screeners' state. The only shared inputs are `StockData` and the SPY benchmark, both provided by the scan orchestrator.

2. **Determinism** — Given the same price data and parameters, SE must produce identical output. No randomness, no network calls, no mutable global state. This property is enforced by Gate 2 (temporal integrity) and Gate 5 (golden regression).

3. **Graceful degradation** — Missing data produces `None` fields, not exceptions. The data policy layer (`policy.py`) evaluates sufficiency upfront. When data is insufficient, SE returns a well-formed payload with null scores and the policy failure recorded in `explain.failed_checks`.

4. **Structural preference** — Among competing pattern candidates with similar scores, SE prefers structural patterns (cup-with-handle, VCP) over event-driven triggers (NR7, first pullback). This reflects Minervini/O'Neill methodology where multi-week structural bases carry more conviction.

5. **Informational flags** — Operational flags (`too_extended`, `breaks_50d_support`, `low_liquidity`, `earnings_soon`) are advisory. They do not affect `setup_ready` or gate evaluation. This keeps the scoring pipeline pure while surfacing contextual risk to the operator.

### Key Decisions

**Decision 1: Permissive gates**

Gates 5–10 pass when their input is `None`. This means a stock with missing RS data or unknown stage still gets scored rather than rejected. Rationale: SE runs on daily price data only (`needs_fundamentals=False`). Many computed fields may be unavailable for recently listed stocks. A strict posture would exclude exactly the growth stocks the methodology targets.

Source: `explain_builder.py:127-179`

Cross-ref: [SE-H1 §3 Setup Ready Gate Map](se_h1_parameter_catalog.md#3-setup-ready-gate-map-10-gates)

**Decision 2: Canonical weights (60/40 + 55/35/10)**

The top-level score weights `SETUP_SCORE_WEIGHTS = (0.60, 0.40)` give quality 50% more influence than readiness. Quality reflects pattern strength — the detector's confidence in what it found. Readiness reflects market context — how close the stock is to triggering. A beautiful pattern in a hostile context is less actionable than a good pattern in a ready context, but quality still dominates because false-positive patterns waste the most operator time.

The per-candidate weights `CANDIDATE_SETUP_SCORE_WEIGHTS = (0.55, 0.35, 0.10)` add a 10% confidence term to break ties between candidates from different detectors.

Source: `config.py:16-18`

Cross-ref: [SE-H1 §2 Score Synthesis Architecture](se_h1_parameter_catalog.md#2-score-synthesis-architecture)

**Decision 3: Deterministic primary selection**

When the aggregator has multiple candidates, it selects the primary by `candidate_score` descending. Ties are broken by structural preference (Decision 4), then by detector execution order. This three-level tiebreaker ensures identical output across runs.

Source: `aggregator.py:250-280`

**Decision 4: Structural preference rule**

The `_STRUCTURAL_TIE_EPSILON = 0.015` constant in `aggregator.py:79` defines "close enough to be a tie." When two candidates' scores differ by ≤ 1.5%, the structural pattern wins. The structural preference order follows the detector execution order defined in `detectors/__init__.py:20-28`:

1. CupWithHandle (structural)
2. ThreeWeeksTight (structural)
3. HighTightFlag (structural)
4. FirstPullback (event-driven)
5. VCP (structural)
6. NR7InsideDay (event-driven)
7. DoubleBottom (stub)

Source: `aggregator.py:79`, `detectors/__init__.py:18-28`

**Decision 5: Informational-only operational flags**

The four operational flags are deliberately decoupled from `derived_ready`. This was a conscious choice: the 10-gate evaluation produces a clean pass/fail signal based on quantitative thresholds. Flags like `earnings_soon` require judgment (some traders buy into earnings, some don't). Mixing judgment calls into the scoring pipeline would make SE less useful as an objective screening tool.

Source: `operational_flags.py:37-91`

Cross-ref: [SE-H4 §6 Explain Payload Reference](se_h4_formula_dictionary_operator_runbook.md#6-explain-payload-reference)

---

## 5. Extension Guardrails

### 5.1 Adding a Detector

1. Create a new module in `analysis/patterns/` (e.g., `ascending_base.py`)
2. Implement the `PatternDetector` interface from `detectors/base.py` — must define `detector_name`, `detect()` returning `PatternDetectorResult`
3. Handle all edge cases in `detect()` — return `DetectorOutcome.ERROR` with detail rather than raising exceptions
4. Add the detector class to `detectors/__init__.py` imports
5. Add an instance to `default_pattern_detectors()` in the desired execution order position
6. Add the detector to `detectors/__init__.py:__all__`
7. Write detector unit tests and add them to Gate 1 in the `Makefile`
8. Generate golden snapshots: `make golden-update`
9. Verify all gates pass: `make gates`
10. Update this document §1 (detector count), §3 (file inventory), and SE-H4 if the detector introduces new field semantics

### 5.2 Adding a Parameter

1. Add the field to `SetupEngineParameters` dataclass in `config.py`
2. Add a corresponding `SetupEngineParameterSpec` to `SETUP_ENGINE_PARAMETER_SPECS` with `min_value`, `max_value`, `unit`, `profile`, and `rationale`
3. Wire the parameter into the consuming module (detector, readiness, or explain_builder)
4. Update [SE-H1](se_h1_parameter_catalog.md) tuning card and parameter detail section
5. Run `make gates` — Gate 1 validates parameter specs, Gate 5 catches output drift

### 5.3 Adding an Output Field

1. Add a `SetupEngineFieldSpec` entry to `SETUP_ENGINE_FIELD_SPECS` in `models.py`
2. Add the field to the `SetupEnginePayload` TypedDict in `models.py`
3. Populate the field in `build_setup_engine_payload()` in `setup_engine_scanner.py`
4. If the field is queryable, update `setup_engine_contract_v1.md` JSON path table
5. Update [SE-H4](se_h4_formula_dictionary_operator_runbook.md) field index and formula section
6. Run `make gates` — Gate 1 validates field specs match payload, Gate 5 catches output changes

### 5.4 Invariants Table

> **Warning**: Do not change these without understanding the downstream implications documented in the referenced sections.

| Invariant | Location | Why it matters |
|-----------|----------|----------------|
| `SETUP_SCORE_WEIGHTS = (0.60, 0.40)` | `config.py:18` | Changes the meaning of every stored `setup_score`. Requires re-scanning all historical data. See [§4 Decision 2](#key-decisions). |
| `CANDIDATE_SETUP_SCORE_WEIGHTS = (0.55, 0.35, 0.10)` | `config.py:16` | Changes primary candidate selection. Shifts which pattern is "best" for multi-pattern stocks. |
| `_STRUCTURAL_TIE_EPSILON = 0.015` | `aggregator.py:79` | Widens or narrows the structural preference window. Too wide → structural bias; too narrow → unstable primary selection. See [§4 Decision 4](#key-decisions). |
| Detector execution order | `detectors/__init__.py:20-28` | Affects tiebreaking in primary selection and the order of `candidates[]` in output. Changing order may change which pattern is primary for tied stocks. |
| `SETUP_ENGINE_FIELD_SPECS` (31 entries) | `models.py:235-484` | Schema contract. Adding fields is safe; removing or renaming fields breaks persistence and frontend queries. See [Setup Engine Contract v1](setup_engine_contract_v1.md). |
| Gate evaluation order (gates 1–10) | `explain_builder.py:96-179` | Gates are evaluated sequentially and all failures are collected. Reordering does not change `derived_ready` but changes the order of `failed_checks[]`, which may affect golden snapshots. |
| `schema_version = "1"` | `models.py` | Must be bumped on any breaking schema change. Frontend and persistence layer use this for compatibility dispatch. |

---

## 6. Quality Gate Reference

### Gate Summary

| Gate | Name | Purpose | Test files | Command |
|-----:|------|---------|------------|---------|
| 1 | Detector correctness | Detectors produce correct outputs, contracts honored, schemas validate | 8 files | `make gate-1` |
| 2 | Temporal integrity | No future-data leakage, data policies enforce sufficiency | 4 files | `make gate-2` |
| 3 | Integration coverage | Round-trip persistence, feature flags, query pipeline, path parity | 6 files | `make gate-3` |
| 4 | Performance baselines | Runtime budget regression (advisory — does not block CI) | 1 file | `make gate-4` |
| 5 | Golden regression | Snapshot-pinned detector, aggregator, and scanner outputs | 3 files | `make gate-5` |

**Total: 22 test files across 5 gates.**

### Test File Assignments (22 files)

| Gate | Test file | What it covers |
|-----:|-----------|---------------|
| 1 | `test_detector_interface_contract.py` | Detectors implement `PatternDetector` correctly |
| 1 | `test_detector_subtasks_c3a_c4a_c6a.py` | Detector subtask acceptance criteria |
| 1 | `test_detector_fixtures_se_g1.py` | Fixture-driven detector output validation |
| 1 | `test_setup_engine_contract.py` | Schema contract compliance |
| 1 | `test_setup_engine_report_schema.py` | Report serialization guards |
| 1 | `test_setup_engine_screener.py` | End-to-end screener integration |
| 1 | `test_setup_engine_parameters.py` | Parameter validation and bounds checking |
| 1 | `test_aggregator_execution_pipeline.py` | Aggregator determinism and primary selection |
| 2 | `test_temporal_integrity_no_lookahead.py` | No future-data leakage in any detector |
| 2 | `test_setup_engine_data_policy.py` | Insufficient-data rejection |
| 2 | `test_setup_engine_score_trace.py` | Score trace formula auditability |
| 2 | `test_setup_engine_readiness.py` | Readiness feature computation correctness |
| 3 | `test_setup_engine_persistence.py` | Round-trip DB write/read |
| 3 | `test_setup_engine_feature_flag.py` | Feature flag enable/disable behavior |
| 3 | `test_setup_engine_query_integration.py` | JSON path queries on SE payload |
| 3 | `test_backfill_setup_engine.py` | Historical backfill correctness |
| 3 | `test_scan_parity.py` | Multi-screener scan path parity |
| 3 | `test_scan_path_parity.py` | Code path equivalence between scan modes |
| 4 | `test_setup_engine_performance.py` | Runtime budget regression |
| 5 | `test_golden_detectors.py` | Pinned detector outputs |
| 5 | `test_golden_aggregator.py` | Pinned aggregator outputs |
| 5 | `test_golden_scanner.py` | Pinned end-to-end scanner outputs |

### Common Gate Commands

```bash
make gates           # Run all 5 gates sequentially
make gate-1          # Run only Gate 1
make gate-check      # Verify every SE test file is assigned to a gate
make golden-update   # Regenerate golden snapshots after intentional output changes
make all             # Full CI: gate-check + all backend gates + frontend lint + frontend tests
```

> **Warning**: After any change that affects SE output (new detector, parameter change, field addition), run `make golden-update` followed by `make gate-5` to update and verify golden snapshots. Do not commit golden snapshot changes without reviewing the diff — unexpected changes indicate a regression.

---

## 7. Backlog Overview

### Deferred Enhancement Summary

| # | Enhancement | Priority | Effort | Status | Key files |
|--:|-------------|----------|--------|--------|-----------|
| 8.1 | Industry-basket RS | High | Medium | Not started | `readiness.py`, `explain_builder.py` |
| 8.2 | Richer earnings integration | Medium | Medium | Not started | `operational_flags.py`, `config.py` |
| 8.3 | Dynamic parameter profiles | Medium | Low | Not started | `config.py` |
| 8.4 | Double Bottom detector | Low | Medium | Stub exists | `detectors/double_bottom.py` |
| 8.5 | Weekly timeframe expansion | Low | High | Not started | All 7 detectors, `technicals.py` |
| 8.6 | Expanded operational flags | Low | Low | Not started | `operational_flags.py` |
| 8.7 | Schema v2 migration | Very Low | High | Not started | `models.py`, `report.py` |
| 8.8 | Cross-detector ensemble scoring | Very Low | High | Not started | `aggregator.py`, `calibration.py` |

### Known Limitations

The current SE v1 has four structural limitations that scope the backlog above:

- **Daily-only timeframe.** All 7 detectors operate on daily OHLCV bars. Weekly patterns (e.g., weekly cup-with-handle) are not detected, even though `technicals.py` already provides a `resample_ohlcv()` helper.
- **RS is stock-vs-SPY only.** The `rs_vs_spy_65d` and `rs_vs_spy_trend_20d` fields compare each stock against the S&P 500 benchmark. There is no industry-group or sector-relative RS, which limits the methodology's ability to find leaders within strong groups (a core O'Neill/Minervini tenet).
- **`double_bottom` returns `not_implemented`.** The detector stub exists at `detectors/double_bottom.py` (71 lines) but returns `DetectorOutcome.NOT_IMPLEMENTED` for every input. It participates in the pipeline without affecting scoring.
- **Earnings context is proximity-only.** The `earnings_soon` operational flag uses a simple date-window check (`0 ≤ days_until_earnings ≤ 21`). It does not consider earnings surprise history, analyst estimates, or post-earnings drift patterns.

---

## 8. Enhancement Details

### 8.1 Industry-Basket RS

**Priority:** High | **Effort:** Medium | **Status:** Not started

Add a `rs_vs_group_65d` field comparing each stock's 65-day performance against its IBD industry group rather than SPY alone. Sector-relative strength is a first-class signal in both O'Neill and Minervini methodologies, and the current RS fields only answer "is this stock outperforming the broad market?"

**Implementation hints:**

- `ibd_groups` and `ibd_group_ranks` tables already exist and are populated by the group rankings pipeline
- Insertion point: `readiness.py:compute_breakout_readiness_features()` — add `group_benchmark_series` alongside existing `benchmark_close`
- Add `rs_vs_group_65d` to `BreakoutReadinessFeatures` (`readiness.py:23-36`), `SETUP_ENGINE_FIELD_SPECS` (`models.py`), and payload (`setup_engine_scanner.py`)
- Consider extending gate 7 (RS leadership) to incorporate group RS

**Dependencies:** Requires reliable group-to-ticker mapping via populated `ibd_groups` table.

**Effort estimate:** ~200 lines across 4 files, plus tests.

### 8.2 Richer Earnings Integration

**Priority:** Medium | **Effort:** Medium | **Status:** Not started

Extend the `earnings_soon` operational flag into richer earnings context (surprise history, estimate revisions, post-earnings drift). The current flag is binary and proximity-based — a stock 2 days before earnings with 10 consecutive beats is very different from one with erratic surprises.

**Implementation hints:**

- Set `needs_earnings_history=True` in `setup_engine_screener.py:64-71` (currently `False`)
- Extend `OperationalFlagInputs` (`operational_flags.py:21-34`) with optional earnings history fields
- New informational flags: `earnings_beat_streak`, `earnings_estimate_revision_positive`, `earnings_gap_risk`
- Keep all earnings flags informational per [§4 Decision 5](#key-decisions); the `stock_fundamentals` table already stores EPS data

**Dependencies:** Requires earnings history data to be reliably fetched. Alpha Vantage free tier is 25 req/day.

**Effort estimate:** ~150 lines across 3 files, plus tests.

### 8.3 Dynamic Parameter Profiles

**Priority:** Medium | **Effort:** Low | **Status:** Not started

Add named parameter presets ("aggressive", "conservative", "momentum") selectable from the UI. `build_setup_engine_parameters(overrides={})` in `config.py:299-315` already accepts arbitrary overrides — the missing piece is a preset-name-to-overrides mapping plus UI wiring.

**Implementation hints:**

- Add `PARAMETER_PROFILES: Dict[str, Dict[str, float]]` to `config.py`
- Call as `build_setup_engine_parameters(overrides=PARAMETER_PROFILES["aggressive"])`
- Frontend: profile selector dropdown; API: `parameter_profile` field in scan request schema

**Dependencies:** None — purely additive.

**Effort estimate:** ~50 lines backend, ~100 lines frontend, plus tests.

### 8.4 Double Bottom Detector

**Priority:** Low | **Effort:** Medium | **Status:** Stub exists

Implement the full double-bottom detection algorithm. The stub at `detectors/double_bottom.py` (71 lines) returns `DetectorOutcome.NOT_IMPLEMENTED` for every input. The execution order position is already reserved.

**Implementation hints:**

- Use `cup_handle.py` (552 lines) as template — both patterns involve two troughs with a middle peak
- Key logic: identify two troughs at approximately the same level, rally between them, breakout above middle peak
- Add parameters to `config.py` (e.g., `double_bottom_trough_tolerance_pct`, `double_bottom_min_depth_pct`)
- Follow the [§5.1 Adding a Detector](#51-adding-a-detector) checklist

**Dependencies:** None.

**Effort estimate:** ~300 lines detector + ~150 lines tests.

### 8.5 Weekly Timeframe Expansion

**Priority:** Low | **Effort:** High | **Status:** Not started

Extend all 7 detectors to operate on weekly OHLCV bars in addition to daily. Weekly patterns (e.g., 6-month cup-with-handle) carry more conviction in institutional-grade analysis.

**Implementation hints:**

- `technicals.py` already provides `resample_ohlcv()` for daily-to-weekly conversion
- `timeframe` field in `SETUP_ENGINE_FIELD_SPECS` already supports `Literal['daily', 'weekly']` — currently always `'daily'`
- Major challenge: parameter values tuned for daily bars may not suit weekly bars — each needs a weekly profile
- Consider parallel daily+weekly detection with candidate merging, or separate scan modes

**Dependencies:** Weekly parameter calibration for all 18 parameters. Benefits from §8.3 (dynamic parameter profiles).

**Effort estimate:** ~500 lines detector mods + ~200 lines profiles + ~300 lines tests.

### 8.6 Expanded Operational Flags

**Priority:** Low | **Effort:** Low | **Status:** Not started

Add new operational flags (`near_major_resistance`, `sector_rotation_headwind`, `extreme_breadth`) and optionally allow hard flags to block readiness via a `hard_flags_block_readiness: bool = False` toggle in `SetupEngineParameters`.

**Implementation hints:**

- Add flags to `compute_operational_flags()` in `operational_flags.py:37-91`
- Optional: wire hard flags into `explain_builder.py` gate evaluation, default off for backward compatibility

**Dependencies:** Some flags (e.g., `sector_rotation_headwind`) require market breadth data not yet in the SE pipeline.

**Effort estimate:** ~60 lines per new flag, plus tests.

### 8.7 Schema v2 Migration

**Priority:** Very Low | **Effort:** High | **Status:** Not started

Design a versioned payload parser supporting both v1 and v2 schemas, enabling breaking field changes without losing historical data. Currently `schema_version = "1"` is baked into every stored payload with no migration path.

**Implementation hints:**

- Add `parse_setup_engine_payload(raw: dict)` dispatcher to `report.py` — reads `schema_version`, delegates to version-specific parsers
- Backfill: migrate stored payloads in-place (one-time) or convert on-read (lazy)
- `models.py` would need `SETUP_ENGINE_FIELD_SPECS_V1` and `V2`; Gate 1 must validate both

**Dependencies:** Implement just-in-time when a breaking schema change is actually required.

**Effort estimate:** ~400 lines migration infrastructure + ~200 lines tests.

### 8.8 Cross-Detector Ensemble Scoring

**Priority:** Very Low | **Effort:** High | **Status:** Not started

Replace single-primary selection with multi-pattern weighted voting. Currently a stock showing both a VCP and NR7 inside day scores the same as one showing only a VCP — the ensemble agreement is not captured.

**Implementation hints:**

- Most architecturally invasive enhancement — rearchitects `aggregator.py` and `calibration.py`
- Replace `_select_primary_candidate()` with ensemble scorer: `ensemble_score = Σ(candidate_score × detector_weight × confidence) / Σ(detector_weight)`
- Requires new `DETECTOR_ENSEMBLE_WEIGHTS` in `config.py`; changes semantics of `setup_score` and `pattern_primary`
- Simpler intermediate step: add `multi_pattern_bonus` field when ≥ 2 candidates detected

**Dependencies:** Should follow §8.7 (schema v2) since it changes score semantics. Requires calibration against historical data.

**Effort estimate:** ~600 lines scoring rearchitecture + ~300 lines tests + recalibration.

---

## 9. Document Cross-Reference Index

All 15 Setup Engine documents in the `docs/setup_engine/` directory:

| ID | Title | File | Role |
|----|-------|------|------|
| SE-A1 | Compatibility Matrix and Integration Risk Audit | `se_a1_compatibility_matrix.md` | Maps SE insertion points across scanner orchestration and query layers |
| SE-A3 | Parameter Governance | `se_a3_parameter_governance.md` | Rationale for parameter dataclass and validation architecture |
| SE-A4 | Data Requirements and Incomplete-Data Policy | `se_a4_data_requirements_policy.md` | Deterministic behavior under missing/short data |
| SE-B1 | Patterns Package Skeleton and Boundaries | `se_b1_patterns_package_skeleton.md` | Package structure and module boundaries |
| SE-B2 | PatternCandidate and Shared Output Schemas | `se_b2_candidate_schema.md` | Canonical `PatternCandidateModel` and shared types |
| SE-B3 | Shared Technical Utilities | `se_b3_technical_utilities.md` | Technical utilities in `technicals.py` |
| SE-B4 | DataFrame Normalization and Validation Guards | `se_b4_dataframe_normalization.md` | Normalization and validation in `normalization.py` |
| SE-B6 | Detector and Aggregator Stub Contracts | `se_b6_detector_stub_contracts.md` | Detector interface contracts for pattern wrappers |
| SE-B7 | SetupEngine Report Schema Types | `se_b7_report_schema_guards.md` | Report schema types and serialization guards |
| SE-C7 | Cross-Detector Normalization and Calibration | `se_c7_cross_detector_calibration.md` | Cross-detector score normalization and confidence calibration |
| SE-H1 | Parameter Catalog and Calibration Report | `se_h1_parameter_catalog.md` | All 18 thresholds, constants, calibration profiles, readiness gates |
| SE-H2 | Staged Rollout Plan | `se_h2_staged_rollout_plan.md` | Feature flag rollout, observability metrics, rollback procedures |
| SE-H3 | Maintainer Handoff and Follow-Up Backlog | `se_h3_maintainer_handoff.md` | This document — orientation guide and deferred enhancement backlog |
| SE-H4 | Formula Dictionary and Operator Runbook | `se_h4_formula_dictionary_operator_runbook.md` | Field-level formulas, pipeline walkthrough, diagnostic procedures |
| — | Setup Engine Contract v1 | `setup_engine_contract_v1.md` | Canonical schema contract for persistence and frontend consumption |

**Documentation series key:**
- **A-series** — Architecture audits and risk assessments
- **B-series** — Blueprints, schemas, and module contracts
- **C-series** — Calibration and normalization specifications
- **H-series** — How-to guides, handbooks, and operational playbooks

---

## 10. Keeping This Document Current

### Update Triggers

| Event | What to update |
|-------|---------------|
| New detector added | §1 dimension table (detector count), §3 file inventory, §5.4 detector execution order |
| New backlog item identified | §7 summary table, §8 new subsection |
| Backlog item completed | §7 status column, §8 subsection status. Remove from backlog when merged and verified. |
| Architecture change (new layer, new file) | §3 file inventory and data-flow diagram |
| New SE document created | §9 cross-reference index |
| Design decision changed | §4 key decisions section — preserve the old rationale as a historical note |
| New quality gate added | §6 gate summary and detail tables |

### Source of Truth

| What | Source |
|------|--------|
| File inventory and line counts | `wc -l` on `backend/app/scanners/setup_engine_*.py` and `backend/app/analysis/patterns/**/*.py` |
| Parameter count and defaults | `config.py:SetupEngineParameters` |
| Field count and specs | `models.py:SETUP_ENGINE_FIELD_SPECS` |
| Gate count and semantics | `explain_builder.py:build_explain_payload()` |
| Operational flag count | `operational_flags.py:compute_operational_flags()` |
| Detector execution order | `detectors/__init__.py:default_pattern_detectors()` |
| Test file gate assignments | `Makefile` gate variables |

### Document Provenance

| Field | Value |
|-------|-------|
| Author | Generated with Claude Code |
| Created | 2026-02-22 |
| SE-H1 version referenced | Current as of commit `1fe0236d` |
| SE-H2 version referenced | Current as of commit `891d3e06` |
| SE-H4 version referenced | Current as of commit `744671aa` |
