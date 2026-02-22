# SE-H4: Formula Dictionary and Operator Runbook

> **Field-level reference** for every Setup Engine output and **operational playbook** for diagnosing, troubleshooting, and maintaining the SE pipeline. Intended for operators running scans, developers debugging scorer behavior, and maintainers onboarding to the codebase.

## Table of Contents

### Part A — Field Dictionary
1. [Quick-Reference Field Index](#1-quick-reference-field-index)
2. [Score & Verdict Fields](#2-score--verdict-fields)
3. [Pattern Fields](#3-pattern-fields)
4. [Readiness Feature Fields](#4-readiness-feature-fields)
5. [Context Fields](#5-context-fields)
6. [Explain Payload Reference](#6-explain-payload-reference)
7. [Candidate Sub-Object Reference](#7-candidate-sub-object-reference)
8. [Pipeline Walkthrough](#8-pipeline-walkthrough)

### Part B — Operator Runbook
9. [Quick Triage Checklist](#9-quick-triage-checklist)
10. [Threshold Tuning Playbook](#10-threshold-tuning-playbook)
11. [Common Failure Modes](#11-common-failure-modes)
12. [Operational Troubleshooting Decision Trees](#12-operational-troubleshooting-decision-trees)
13. [Ownership & Escalation Map](#13-ownership--escalation-map)
14. [Keeping This Document Current](#14-keeping-this-document-current)

### Related Design Docs

- [SE-H1: Parameter Catalog](se_h1_parameter_catalog.md) — Threshold defaults, bounds, guardrails, calibration profiles
- [SE-H2: Staged Rollout Plan](se_h2_staged_rollout_plan.md) — Rollout phases, feature flags, rollback procedures
- [SE-A3: Parameter Governance](se_a3_parameter_governance.md) — Parameter dataclass and validation architecture
- [SE-A4: Data Requirements Policy](se_a4_data_requirements_policy.md) — Data sufficiency semantics
- [SE-C7: Cross-Detector Calibration](se_c7_cross_detector_calibration.md) — Normalization algebra and rank-score derivation

### Canonical Source Files

| File | Content |
|------|---------|
| `backend/app/analysis/patterns/models.py:235-484` | All 31 `SETUP_ENGINE_FIELD_SPECS` entries |
| `backend/app/analysis/patterns/readiness.py` | 10 readiness feature formulas |
| `backend/app/analysis/patterns/explain_builder.py` | 10-gate evaluation, check string enumeration |
| `backend/app/analysis/patterns/operational_flags.py` | 4 operational flag conditions |
| `backend/app/analysis/patterns/trace.py` | Score trace formula strings |
| `backend/app/analysis/patterns/config.py` | Default parameters, weights |
| `backend/app/analysis/patterns/calibration.py` | Calibration profiles, rank score |
| `backend/app/analysis/patterns/aggregator.py` | Primary selection algorithm |
| `backend/app/analysis/patterns/policy.py` | Data requirements and policy evaluation |
| `backend/app/scanners/setup_engine_scanner.py` | Score synthesis, payload assembly |
| `backend/app/scanners/setup_engine_screener.py` | Context computation, rating rules |
| `backend/scripts/backfill_setup_engine.py` | Backfill operations |

---

# Part A — Field Dictionary

## 1. Quick-Reference Field Index

All 31 fields from `SETUP_ENGINE_FIELD_SPECS` (`models.py:235-484`):

| # | Field | Type | Unit | Nullable | Short Formula | Section |
|---|-------|------|------|----------|---------------|---------|
| 1 | `schema_version` | `str` | — | No | Static `"v1"` | (metadata) |
| 2 | `timeframe` | `Literal['daily','weekly']` | — | No | Set by scanner | (metadata) |
| 3 | `setup_score` | `float` | pct | Yes | `0.60 × quality + 0.40 × readiness` | [§2](#2-score--verdict-fields) |
| 4 | `quality_score` | `float` | pct | Yes | Calibrated from primary candidate | [§2](#2-score--verdict-fields) |
| 5 | `readiness_score` | `float` | pct | Yes | Calibrated from primary candidate | [§2](#2-score--verdict-fields) |
| 6 | `setup_ready` | `bool` | — | No | All 10 gates pass AND readiness ≥ threshold | [§2](#2-score--verdict-fields) |
| 7 | `pattern_primary` | `str` | — | Yes | Highest `aggregation_rank_score` candidate | [§3](#3-pattern-fields) |
| 8 | `pattern_confidence` | `float` | pct | Yes | Calibrated confidence of primary (0–100) | [§3](#3-pattern-fields) |
| 9 | `pivot_price` | `float` | price | Yes | From primary candidate's detector | [§3](#3-pattern-fields) |
| 10 | `pivot_type` | `str` | — | Yes | Pivot family (breakout, pullback, reclaim) | [§3](#3-pattern-fields) |
| 11 | `pivot_date` | `str(YYYY-MM-DD)` | — | Yes | Date associated with pivot_price | [§3](#3-pattern-fields) |
| 12 | `distance_to_pivot_pct` | `float` | pct | Yes | `100 × (close − pivot) / pivot` | [§4](#4-readiness-feature-fields) |
| 13 | `atr14_pct` | `float` | pct | Yes | `100 × ATR(14) / close` | [§4](#4-readiness-feature-fields) |
| 14 | `atr14_pct_trend` | `float` | pct | Yes | `slope(atr14_pct, window=20)` | [§4](#4-readiness-feature-fields) |
| 15 | `bb_width_pct` | `float` | pct | Yes | `100 × (bb_upper − bb_lower) / bb_mid` | [§4](#4-readiness-feature-fields) |
| 16 | `bb_width_pctile_252` | `float` | pct | Yes | `pct_rank(bb_width_pct, window=252)` | [§4](#4-readiness-feature-fields) |
| 17 | `volume_vs_50d` | `float` | ratio | Yes | `volume / SMA(volume, 50)` | [§4](#4-readiness-feature-fields) |
| 18 | `rs` | `float` | ratio | Yes | `close / benchmark_close` | [§4](#4-readiness-feature-fields) |
| 19 | `rs_line_new_high` | `bool` | — | No | `rs_current ≥ max(rs[−252:])` | [§4](#4-readiness-feature-fields) |
| 20 | `rs_vs_spy_65d` | `float` | pct | Yes | `100 × (rs / rs[−65] − 1)` | [§4](#4-readiness-feature-fields) |
| 21 | `rs_vs_spy_trend_20d` | `float` | ratio | Yes | `slope(rs, window=20)` | [§4](#4-readiness-feature-fields) |
| 22 | `stage` | `int` | — | Yes | Weinstein stage 1–4 via `quick_stage_check()` | [§5](#5-context-fields) |
| 23 | `ma_alignment_score` | `float` | pct | Yes | Minervini MA alignment (50>150>200) | [§5](#5-context-fields) |
| 24 | `rs_rating` | `float` | pct | Yes | Multi-period weighted RS (50 = matching SPY) | [§5](#5-context-fields) |
| 25 | `candidates` | `list[PatternCandidate]` | — | No | All detector candidates (calibrated) | [§7](#7-candidate-sub-object-reference) |
| 26 | `explain` | `SetupEngineExplain` | — | No | Structured pass/fail checks | [§6](#6-explain-payload-reference) |
| 27 | `explain.passed_checks` | `list[str]` | — | No | Checks passed by the setup | [§6](#6-explain-payload-reference) |
| 28 | `explain.failed_checks` | `list[str]` | — | No | Checks failed by the setup | [§6](#6-explain-payload-reference) |
| 29 | `explain.key_levels` | `dict[str, float\|None]` | price | No | Named price levels (pivot, support, etc.) | [§6](#6-explain-payload-reference) |
| 30 | `explain.invalidation_flags` | `list[str]` | — | No | Operational risk warnings | [§6](#6-explain-payload-reference) |
| 31 | `explain.score_trace` | `ScoreTrace` | — | Yes | Per-field calculation trace (opt-in) | [§6](#score-trace-format) |

**Non-computed metadata fields:**
- `schema_version`: Always `"v1"` (from `SETUP_ENGINE_DEFAULT_SCHEMA_VERSION` in `models.py:15`). Used for future schema migration gates.
- `timeframe`: Set to `"daily"` by the screener (`setup_engine_screener.py:169`). Allowed values: `{"daily", "weekly"}`.

The `explain.score_trace` field (#31) is present only when explicitly opted in via `include_score_trace=True`. See [§6 Score Trace Format](#score-trace-format) for details.

---

## 2. Score & Verdict Fields

### `setup_score`

**Formula:**

```python
setup_score = clamp(0.60 × quality_score + 0.40 × readiness_score, 0, 100)
```

**Source:** `setup_engine_scanner.py:324-329`

**Unit/Range:** 0–100 (pct), nullable

**Interpretation:** Composite measure of how good the pattern is (quality) and how ready it is to break out (readiness). The 60/40 weighting prioritizes pattern quality over timing readiness, reflecting the view that a well-formed base is more predictive than optimal timing.

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| `quality_score = None` | `setup_score = None` | Cannot synthesize without both components |
| `readiness_score = None` | `setup_score = None` | Cannot synthesize without both components |
| Both present | Computed | Normal path |
| `data_policy = "insufficient"` | `setup_score = None` | All scores nulled on insufficient data |

**Consumed by:** Gate 1 (`setup_score_ok`), rating rules, ScreenerResult score

**SE-H1 cross-ref:** [§2 Score Synthesis Architecture](se_h1_parameter_catalog.md#2-score-synthesis-architecture), [§4.4 Score Gates (`setup_score_min_pct`)](se_h1_parameter_catalog.md#44-score-gates)

---

### `quality_score`

**Formula:**

```python
# Extracted from the primary candidate after cross-detector calibration:
quality_norm = (raw_quality - profile.quality_min) / (profile.quality_max - profile.quality_min)
quality_score = clamp(quality_norm × 100, 0, 100)
```

**Source:** `calibration.py:137-149` (normalization), `setup_engine_scanner.py:235-243` (extraction)

**Unit/Range:** 0–100 (pct), nullable

**Interpretation:** Measures how well the detected pattern conforms to the ideal geometric shape. A score of 80 means the pattern matches ~80% of the structural criteria for its pattern type. Calibrated across detector families so that a quality_score of 70 from a cup-with-handle carries the same meaning as 70 from a VCP.

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| No candidates | `quality_score = None` | No pattern detected |
| Primary candidate missing quality | `quality_score = None` | Detector did not compute quality |
| `data_policy = "insufficient"` | `quality_score = None` | Pipeline nulled |

**Consumed by:** Gate 2 (`quality_floor_ok`), `setup_score` synthesis

**SE-H1 cross-ref:** [§4.4 Score Gates (`quality_score_min_pct`)](se_h1_parameter_catalog.md#44-score-gates), [§9 Cross-Detector Calibration Profiles](se_h1_parameter_catalog.md#9-cross-detector-calibration-profiles)

---

### `readiness_score`

**Formula:**

```python
# Extracted from the primary candidate after cross-detector calibration:
readiness_norm = (raw_readiness - profile.readiness_min) / (profile.readiness_max - profile.readiness_min)
readiness_score = clamp(readiness_norm × 100, 0, 100)
```

**Source:** `calibration.py:138-153` (normalization), `setup_engine_scanner.py:235-243` (extraction)

**Unit/Range:** 0–100 (pct), nullable

**Interpretation:** Measures how close the stock is to an actionable breakout point. Incorporates volume confirmation, proximity to pivot, trend resumption signals, and other detector-specific timing factors. Higher values indicate the breakout is more imminent.

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| No candidates | `readiness_score = None` | No pattern detected |
| Primary candidate missing readiness | `readiness_score = None` | Detector did not compute readiness |
| `data_policy = "insufficient"` | `readiness_score = None` | Pipeline nulled |

**Consumed by:** Gate 3 (`readiness_floor_ok`), `setup_score` synthesis

**SE-H1 cross-ref:** [§4.4 Score Gates (`readiness_score_ready_min_pct`)](se_h1_parameter_catalog.md#44-score-gates), [§9 Cross-Detector Calibration Profiles](se_h1_parameter_catalog.md#9-cross-detector-calibration-profiles)

---

### `setup_ready`

**Formula:**

```python
# Derived when not explicitly set:
setup_ready = (len(explain.failed_checks) == 0)
# i.e., ALL 10 gates pass AND zero pre-existing failures
```

**Source:** `explain_builder.py:182` (derivation), `setup_engine_scanner.py:367` (assignment)

**Unit/Range:** boolean, not nullable

**Interpretation:** The **final verdict** of the scoring pipeline. `True` means the stock has a high-quality pattern, is near its breakout point, and passes all readiness gates. This is the field that determines whether the stock appears as a "Buy" or "Strong Buy" candidate.

**Critical behavior:** Operational flags (`too_extended`, `breaks_50d_support`, `low_liquidity`, `earnings_soon`) do **NOT** affect `setup_ready`. They are informational warnings only. A stock can be `setup_ready = True` while carrying a `too_extended` operational flag.

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| `setup_score = None` | `setup_ready = False` | Insufficient data skips all gates |
| Any single gate fails | `setup_ready = False` | All-pass requirement |
| All 10 gates pass + zero pre-existing failures | `setup_ready = True` | Full qualification |
| Explicit `setup_ready` parameter provided | Uses explicit value | Override path (rare) |

**Consumed by:** Rating rules (`calculate_rating()`), ScreenerResult `passes` field

**SE-H1 cross-ref:** [§3 Setup Ready Gate Map](se_h1_parameter_catalog.md#3-setup-ready-gate-map-10-gates)

---

## 3. Pattern Fields

### `pattern_primary`

**Formula:**

```python
# Select candidate with highest aggregation_rank_score:
# rank_score = 0.55 × confidence + 0.25 × (quality/100) + 0.20 × (readiness/100)
# With structural tie-break: if top candidate is a trigger pattern and
# a structural candidate is within 0.015 rank_score, prefer structural.
primary = max(candidates, key=aggregation_rank_score)
```

**Source:** `aggregator.py:244-283` (selection), `calibration.py:239-260` (rank score)

**Unit/Range:** string, nullable

**Interpretation:** The pattern type of the highest-ranked candidate. Possible values include: `vcp`, `three_weeks_tight`, `high_tight_flag`, `cup_with_handle`, `first_pullback`, `nr7_inside_day`, `double_bottom` (stub — always returns `not_implemented`).

**Primary Candidate Selection Algorithm:**

1. Filter candidates to those with `confidence_pct ≥ pattern_confidence_min_pct` (default: 55%)
2. If no candidates qualify, use all candidates as fallback pool (sets `primary_pattern_below_confidence_floor` check)
3. Rank by `aggregation_rank_score` with tie-break key: `(rank_score, confidence, quality+readiness, -|distance_to_pivot|, pattern_name, source_detector, -index)`
4. **Structural tie-break**: If the top candidate is a trigger pattern (`nr7_inside_day`, `first_pullback`) and a structural candidate is within `_STRUCTURAL_TIE_EPSILON = 0.015` rank_score, the structural candidate wins

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| No candidates from any detector | `pattern_primary = None` | `no_primary_pattern` check fails |
| All candidates below confidence floor | Uses best candidate anyway | Fallback selection; `primary_pattern_below_confidence_floor` flagged |
| Tie between trigger and structural | Structural wins | Structural tie-break within epsilon |
| `data_policy = "insufficient"` | `pattern_primary = None` | Candidates cleared |

---

### `pattern_confidence`

**Formula:**

```python
# From primary candidate's calibrated confidence_pct (0-100 scale):
pattern_confidence = primary_candidate["confidence_pct"]
```

**Source:** `aggregator.py:177`

**Unit/Range:** 0–100 (pct), nullable

**Interpretation:** How certain the detector is that this pattern is genuine. Calibrated across detector families via confidence normalization and bias adjustment. Higher values indicate stronger structural evidence for the pattern classification.

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| No primary candidate | `None` | No pattern detected |
| Detector returns no confidence | `None` | Missing upstream value |

---

### Per-Detector Pivot Derivation

Each detector produces its own `pivot_price`, `pivot_type`, and `pivot_date`. The primary candidate's pivot values become the top-level fields:

| Detector | `pivot_price` Source | `pivot_type` | Notes |
|----------|---------------------|--------------|-------|
| **vcp** | Right-side high of contraction | `breakout` | Legacy VCP detector's `buy_point` |
| **three_weeks_tight** | Highest close during tight weeks | `breakout` | Conservative: uses close, not high |
| **high_tight_flag** | Flag high (top of consolidation) | `breakout` | After ≥100% pole advance |
| **cup_with_handle** | Handle high (or right lip if no handle) | `breakout` | Classic O'Neil pivot point |
| **first_pullback** | Prior swing high before pullback | `pullback` | Resumption entry after MA touch |
| **nr7_inside_day** | Prior day's high | `breakout` | Trigger bar reference |
| **double_bottom** | (stub) | — | Returns `not_implemented` |

**Source:** Individual detector modules in `backend/app/analysis/patterns/`

---

### `pivot_price`, `pivot_type`, `pivot_date`

**Source:** `aggregator.py:176-180` (from primary candidate)

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `pivot_price` | `float` | Yes | The canonical buy point / action price for the primary pattern |
| `pivot_type` | `str` | Yes | Pivot family: `breakout`, `pullback`, `reclaim` |
| `pivot_date` | `str(YYYY-MM-DD)` | Yes | Date the pivot was established (detector-specific) |

**Edge cases:** All three are `None` when no primary candidate is selected or data is insufficient. `pivot_date` is always normalized to ISO-8601 format via `normalize_iso_date()`.

---

## 4. Readiness Feature Fields

All 10 readiness features are computed in `readiness.py:82-219` by the `_compute_readiness_core()` function. They measure how close a stock is to breaking out, independent of the pattern type.

### Computation Windows

| Parameter | Default | Used By |
|-----------|---------|---------|
| `atr_period` | 14 | `atr14_pct`, `atr14_pct_trend` |
| `trend_lookback` | 20 | `atr14_pct_trend`, `rs_vs_spy_trend_20d` |
| `bb_window` | 20 | `bb_width_pct` |
| `bb_pctile_window` | 252 | `bb_width_pctile_252` |
| `volume_sma_window` | 50 | `volume_vs_50d` |
| `rs_lookback` | 252 | `rs_line_new_high` |
| `rs_vs_spy_window` | 65 | `rs_vs_spy_65d` |

**SE-H1 cross-ref:** [§11 Data Requirements & Feature Computation](se_h1_parameter_catalog.md#11-data-requirements--feature-computation)

---

### `distance_to_pivot_pct`

**Formula:**

```python
distance_to_pivot_pct = 100 × (close - pivot_price) / pivot_price
```

**Source:** `readiness.py:114-116`

**Unit/Range:** pct (unbounded, typically −20 to +20), nullable

**Interpretation:**

| Range | Meaning |
|-------|---------|
| `< −5` | Well below pivot; base still forming or broken |
| `−5 to −2` | Approaching pivot from below; building cause |
| **`−2 to +3`** | **Early zone (Gate 4 default window)** — actionable range |
| `+3 to +10` | Extended past pivot; late entry risk |
| `> +10` | Over-extended; `too_extended` flag triggers |

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| `pivot_price = None` | `None` | No pivot to measure against |
| `pivot_price = 0` | `None` | Division guard |
| `close = None` | `None` | No valid close price |

**Consumed by:** Gate 4 (`in_early_zone`), `too_extended` operational flag

**SE-H1 cross-ref:** [§4.2 Breakout Zone Window](se_h1_parameter_catalog.md#42-breakout-zone-window)

---

### `atr14_pct`

**Formula:**

```python
atr14_pct = 100 × ATR(14, method="wilder") / close
```

**Source:** `readiness.py:118-127`

**Unit/Range:** pct (typically 1–15), nullable

**Interpretation:**

| Range | Meaning |
|-------|---------|
| `< 3` | Tight base; very low volatility — classic squeeze |
| `3 to 6` | Normal volatility for a constructive base |
| `6 to 8` | Elevated; still within Gate 5 default limit |
| **`> 8`** | **Gate 5 fails** — disorderly price action |
| `> 15` | Extreme volatility; typically climax/distribution behavior |

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| `close = 0` | `None` | Division guard |
| Fewer than 14 bars | `None` | ATR requires minimum data |
| `close = None` | `None` | No valid latest price |

**Consumed by:** Gate 5 (`atr14_within_limit`)

**SE-H1 cross-ref:** [§4.5 Supplemental Readiness Controls (`atr14_pct_max_for_ready`)](se_h1_parameter_catalog.md#45-supplemental-readiness-controls)

---

### `atr14_pct_trend`

**Formula:**

```python
atr14_pct_trend = slope(atr14_pct_series, window=20)
```

**Source:** `readiness.py:129-130`

**Unit/Range:** pct (small values, typically −0.5 to +0.5), nullable

**Interpretation:**

| Range | Meaning |
|-------|---------|
| `< −0.1` | Volatility contracting — bullish for breakout setups |
| `−0.1 to +0.1` | Stable volatility |
| `> +0.1` | Volatility expanding — may indicate breakout in progress or breakdown |

**Edge Cases:** `None` when `atr14_pct_series` has fewer than 20 valid observations.

**Consumed by:** Not directly gated; informational for score trace and operator analysis.

---

### `bb_width_pct`

**Formula:**

```python
bb_width_pct = 100 × (bb_upper - bb_lower) / bb_mid
# where bb_upper, bb_lower, bb_mid = bollinger_bands(close, window=20, stddev=2.0)
```

**Source:** `readiness.py:132-137`

**Unit/Range:** pct (typically 2–30), nullable

**Interpretation:**

| Range | Meaning |
|-------|---------|
| `< 5` | Extremely tight bands — strong squeeze |
| `5 to 10` | Normal tight consolidation |
| `10 to 20` | Moderate width; orderly trend |
| `> 20` | Wide bands; elevated volatility |

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| `bb_mid = 0` | `None` | Division by zero guard (via `_safe_divide`) |
| Fewer than 20 bars | `None` | BB requires full window |

**Consumed by:** `bb_width_pctile_252` computation. Not directly gated at top-level but used by detector-internal squeeze logic.

---

### `bb_width_pctile_252`

**Formula:**

```python
bb_width_pctile_252 = pct_rank(bb_width_pct[-1], bb_width_pct[-252:])
# Returns 0-100: what percentile the current BB width sits at over ~1 year
```

**Source:** `readiness.py:138-140`

**Unit/Range:** 0–100 (pct), nullable

**Interpretation:**

| Range | Meaning |
|-------|---------|
| `< 10` | Historical squeeze — current width is near 1-year low |
| `10 to 20` | Strict squeeze zone (below `squeeze_bb_width_pctile_max_strict` default) |
| `20 to 35` | Relaxed squeeze zone (below `squeeze_bb_width_pctile_max_relaxed` default) |
| `35 to 60` | Normal volatility relative to recent history |
| `> 60` | Above-average volatility; not in squeeze |

**Edge Cases:** `None` when fewer than 252 observations for percentile ranking. Requires full year of BB width history.

**Consumed by:** Detector-internal squeeze evaluation (3WT, VCP). Not directly gated at top-level.

**SE-H1 cross-ref:** [§4.3 Volatility Squeeze](se_h1_parameter_catalog.md#43-volatility-squeeze-bollinger-width-percentile)

---

### `volume_vs_50d`

**Formula:**

```python
volume_vs_50d = volume / SMA(volume, 50)
```

**Source:** `readiness.py:142-143`

**Unit/Range:** ratio (typically 0.2–5.0), nullable

**Interpretation:**

| Range | Meaning |
|-------|---------|
| `< 0.5` | Very low volume — potential holiday/illiquid session |
| `0.5 to 0.8` | Below average; quiet accumulation possible |
| `0.8 to 1.0` | Slightly below average |
| **`≥ 1.0`** | **Gate 6 passes** — at or above average volume |
| `1.5 to 3.0` | Elevated; institutional interest likely |
| `> 3.0` | Unusual volume; could be breakout or capitulation |

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| `SMA(volume, 50) = 0` | `None` | Division guard |
| Fewer than 50 volume bars | `None` | SMA requires full window |

**Consumed by:** Gate 6 (`volume_sufficient`)

**SE-H1 cross-ref:** [§4.5 Supplemental Readiness Controls (`volume_vs_50d_min_for_ready`)](se_h1_parameter_catalog.md#45-supplemental-readiness-controls)

---

### `rs`

**Formula:**

```python
rs = close / benchmark_close  # benchmark = SPY
```

**Source:** `readiness.py:167-169`

**Unit/Range:** ratio (typically 0.001–100+), nullable

**Interpretation:** Raw relative-strength line value. Trends up when the stock outperforms SPY, trends down when underperforming. The absolute value is meaningless — only the direction matters. Used as the basis for `rs_line_new_high`, `rs_vs_spy_65d`, and `rs_vs_spy_trend_20d`.

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| No benchmark data | `None` | RS requires SPY close series |
| `benchmark_close = 0` | `None` | Division guard |

**Consumed by:** Derived RS fields below. Not directly gated.

---

### `rs_line_new_high`

**Formula:**

```python
rs_line_new_high = (rs_current >= max(rs[-252:]) - 1e-12)
# Epsilon tolerance of 1e-12 for floating-point comparison
```

**Source:** `readiness.py:180-183`

**Unit/Range:** boolean, not nullable (defaults to `False`)

**Interpretation:** `True` when the stock's RS line is at or above its 252-session (1-year) high. A new RS high confirms the stock is a genuine leader — it is outperforming SPY by the most it has in the past year. This is one of O'Neil's key buy criteria.

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| No benchmark data | `False` | Cannot compute RS line |
| Fewer than 252 RS values | Uses available tail | `rs_tail = rs_series.dropna().tail(rs_lookback)` |
| RS series empty | `False` | No data to compare |

**Consumed by:** Gate 7 (`rs_leadership_ok`) — passes if `rs_line_new_high = True` OR `rs_vs_spy_65d > 0`

---

### `rs_vs_spy_65d`

**Formula:**

```python
rs_vs_spy_65d = 100 × (rs / rs[-65] - 1)
```

**Source:** `readiness.py:173-175`

**Unit/Range:** pct (typically −30 to +30), nullable

**Interpretation:**

| Range | Meaning |
|-------|---------|
| `< −10` | Significant relative underperformance over 3 months |
| `−10 to 0` | Slight underperformance |
| **`> 0`** | **Outperforming SPY over 65 sessions** — Gate 7 passes |
| `> 10` | Strong relative outperformance |

**Edge Cases:** `None` when fewer than 65 sessions of RS history, or when benchmark is unavailable.

**Consumed by:** Gate 7 (`rs_leadership_ok`)

---

### `rs_vs_spy_trend_20d`

**Formula:**

```python
rs_vs_spy_trend_20d = slope(rs_series, window=20)
```

**Source:** `readiness.py:176-178`

**Unit/Range:** ratio (small values), nullable

**Interpretation:** The short-term direction of the RS line. Positive values indicate improving relative strength. Not directly gated — informational for operators and score trace.

**Edge Cases:** `None` when fewer than 20 valid RS observations.

**Consumed by:** Not directly gated; informational.

---

## 5. Context Fields

These three fields are computed in the screener's `_compute_context()` method (`setup_engine_screener.py:297-351`), **not** in the analysis layer. They provide market-structure context used by Gates 8–10.

### `stage`

**Formula:**

```python
stage = quick_stage_check(current_price, ma_50, ma_150, ma_200, ma_200_month_ago)
```

**Source:** `setup_engine_screener.py:326-329`

**Unit/Range:** integer 1–4, nullable

**Interpretation (Weinstein Stage Analysis):**

| Stage | MA Structure | Market Phase |
|-------|-------------|--------------|
| 1 | Price near MAs, MAs flattening | **Basing** — accumulation zone |
| 2 | Price > MAs, MAs rising, 50 > 150 > 200 | **Advancing** — uptrend, ideal for breakouts |
| 3 | Price falls through MAs, MAs flattening | **Topping** — distribution zone |
| 4 | Price < MAs, MAs declining | **Declining** — downtrend |

**Edge Cases:**

| Input | Output | Why |
|-------|--------|-----|
| Any MA is `NaN` (< 200 bars) | `None` | All context fields nulled |
| `data_policy = "insufficient"` | `None` | Pipeline nulled |

**Consumed by:** Gate 8 (`stage_ok`). Stage 1 and 2 pass. Stage 3 and 4 fail. `None` passes (semi-permissive).

---

### `ma_alignment_score`

**Formula:**

```python
# Uses MovingAverageAnalyzer.comprehensive_ma_analysis():
# Checks: price > 50 > 150 > 200 MA, 200 MA rising, price 30%+ above 52w low,
# price within 25% of 52w high, etc.
ma_alignment_score = ma_result["minervini_ma_score"]
```

**Source:** `setup_engine_screener.py:331-335`

**Unit/Range:** 0–100 (pct), nullable

**Interpretation:** Measures how well the stock's moving average structure matches Minervini's "Template" criteria: price > 50-day MA > 150-day MA > 200-day MA, with the 200-day MA trending up. A score of 100 means perfect Minervini alignment.

**Edge Cases:** `None` when any required MA is `NaN`.

**Consumed by:** Gate 9 (`ma_alignment_ok`)

**SE-H1 cross-ref:** [§4.6 Context Gates (`context_ma_alignment_min_pct`)](se_h1_parameter_catalog.md#46-context-gates)

---

### `rs_rating`

**Formula:**

```python
# Uses RelativeStrengthCalculator.calculate_rs_rating():
# Multi-period weighted performance vs SPY benchmark
# Linear scale where 50 = matching SPY, >50 = outperforming
rs_rating = rs_result["rs_rating"]
```

**Source:** `setup_engine_screener.py:337-343`

**Unit/Range:** 0–100 (pct), nullable

**Interpretation:** A composite relative-strength rating comparing the stock's performance to SPY over multiple time periods. A score of 50 means the stock is matching SPY on a blended basis; scores above 50 indicate outperformance, below 50 indicate underperformance. Unlike `rs_vs_spy_65d` (single window), this uses multiple weighted periods.

**Edge Cases:** `None` when SPY benchmark data is unavailable or empty.

**Consumed by:** Gate 10 (`rs_rating_ok`)

**SE-H1 cross-ref:** [§4.6 Context Gates (`context_rs_rating_min`)](se_h1_parameter_catalog.md#46-context-gates)

---

## 6. Explain Payload Reference

The `explain` sub-object provides human-readable gate results, key price levels, operational warnings, and optional score trace. Built by `build_explain_payload()` in `explain_builder.py`.

### Complete Check String Enumeration

#### Gate Check Strings

| Gate | Pass Check | Fail Check | Source |
|------|-----------|------------|--------|
| 1 | `setup_score_ok` | `setup_score_below_threshold` | `explain_builder.py:97-100` |
| 2 | `quality_floor_ok` | `quality_below_threshold` | `explain_builder.py:103-106` |
| 3 | `readiness_floor_ok` | `readiness_below_threshold` | `explain_builder.py:109-112` |
| 4 | `in_early_zone` | `outside_early_zone` | `explain_builder.py:115-125` |
| 5 | `atr14_within_limit` | `atr14_pct_exceeds_limit` | `explain_builder.py:128-134` |
| 6 | `volume_sufficient` | `volume_below_minimum` | `explain_builder.py:137-143` |
| 7 | `rs_leadership_ok` | `rs_leadership_insufficient` | `explain_builder.py:146-153` |
| 8 | `stage_ok` | `stage_not_ok` | `explain_builder.py:156-161` |
| 9 | `ma_alignment_ok` | `ma_alignment_insufficient` | `explain_builder.py:164-170` |
| 10 | `rs_rating_ok` | `rs_rating_insufficient` | `explain_builder.py:173-179` |

#### Aggregator Check Strings

| Check | Condition | Source |
|-------|-----------|--------|
| `cross_detector_calibration_applied` | Calibration ran on candidates | `aggregator.py:159` |
| `detector_pipeline_executed` | At least one detector ran | `aggregator.py:161` |
| `primary_pattern_selected` | Primary candidate above confidence floor | `aggregator.py:170` |
| `primary_pattern_fallback_selected` | Primary candidate below confidence floor (fallback) | `aggregator.py:168` |
| `primary_pattern_below_confidence_floor` | All candidates below confidence floor | `aggregator.py:167` |
| `primary_pattern_structural_tie_break_applied` | Structural candidate preferred over trigger | `aggregator.py:172` |
| `no_primary_pattern` | No candidates at all | `aggregator.py:164` |
| `{detector_name}:error` | Detector raised an exception | `aggregator.py:128` |

#### Policy Check Strings

| Check | Condition | Source |
|-------|-----------|--------|
| `insufficient_data` | Data policy status = `insufficient` | `policy.py:125-126` |
| `insufficient_daily_bars` | Daily bars < 252 | `policy.py:91` |
| `insufficient_weekly_bars` | Weekly bars < 52 | `policy.py:92` |
| `insufficient_benchmark_bars` | Benchmark bars < 252 (when not degradable) | `policy.py:99` |

#### Detector-Specific Check Strings

Individual detectors may emit their own passed/failed checks (e.g., `tight_band_ok`, `breakout_volume_unconfirmed`). These are detector-internal and passed through to the explain payload unchanged. The `double_bottom` stub detector emits `not_implemented` as its outcome — this is intentionally reserved for future expansion, not a bug.

### Check String → Gate → SE-H1 Parameter Mapping

| Check String | Gate # | SE-H1 Parameter |
|-------------|--------|-----------------|
| `setup_score_ok` / `setup_score_below_threshold` | 1 | [`setup_score_min_pct`](se_h1_parameter_catalog.md#44-score-gates) |
| `quality_floor_ok` / `quality_below_threshold` | 2 | [`quality_score_min_pct`](se_h1_parameter_catalog.md#44-score-gates) |
| `readiness_floor_ok` / `readiness_below_threshold` | 3 | [`readiness_score_ready_min_pct`](se_h1_parameter_catalog.md#44-score-gates) |
| `in_early_zone` / `outside_early_zone` | 4 | [`early_zone_distance_to_pivot_pct_min/max`](se_h1_parameter_catalog.md#42-breakout-zone-window) |
| `atr14_within_limit` / `atr14_pct_exceeds_limit` | 5 | [`atr14_pct_max_for_ready`](se_h1_parameter_catalog.md#45-supplemental-readiness-controls) |
| `volume_sufficient` / `volume_below_minimum` | 6 | [`volume_vs_50d_min_for_ready`](se_h1_parameter_catalog.md#45-supplemental-readiness-controls) |
| `rs_leadership_ok` / `rs_leadership_insufficient` | 7 | (implicit: `rs_vs_spy_65d > 0` or `rs_line_new_high`) |
| `stage_ok` / `stage_not_ok` | 8 | (implicit: `stage ∈ {1, 2}`) |
| `ma_alignment_ok` / `ma_alignment_insufficient` | 9 | [`context_ma_alignment_min_pct`](se_h1_parameter_catalog.md#46-context-gates) |
| `rs_rating_ok` / `rs_rating_insufficient` | 10 | [`context_rs_rating_min`](se_h1_parameter_catalog.md#46-context-gates) |

### Operational Flag Codes

Four operational flags from `operational_flags.py:37-91`. These are **informational only** — they do NOT affect `setup_ready` or `derived_ready`.

| Flag Code | `is_hard` | Trigger Condition | SE-H1 Parameter |
|-----------|-----------|-------------------|-----------------|
| `too_extended` | `False` (soft) | `distance_to_pivot_pct > 10.0` | [`too_extended_pivot_distance_pct`](se_h1_parameter_catalog.md#47-operational-flags-informational) |
| `breaks_50d_support` | **`True`** (hard) | `price < MA(50) × (1 − cushion/100)` | [`breaks_50d_support_cushion_pct`](se_h1_parameter_catalog.md#47-operational-flags-informational) |
| `low_liquidity` | `False` (soft) | `ADTV(50d, USD) < 1,000,000` | [`low_liquidity_adtv_min_usd`](se_h1_parameter_catalog.md#47-operational-flags-informational) |
| `earnings_soon` | `False` (soft) | `0 ≤ days_until_earnings ≤ 21` | [`earnings_soon_window_days`](se_h1_parameter_catalog.md#47-operational-flags-informational) |

Data policy flags are also emitted as invalidation flags:
- `data_policy:degraded` — Missing benchmark or incomplete week
- `data_policy:insufficient` — Critical data missing
- Plus the specific reasons: `missing_or_short_benchmark_history`, `current_week_incomplete_exclude_from_weekly`, etc.

### Score Trace Format

The optional `explain.score_trace` field (from `trace.py`) provides per-field calculation auditability. Each entry is a `FieldTrace` TypedDict:

```python
FieldTrace = {
    "formula": str,        # Human-readable formula string
    "inputs": dict,        # Named input values (rounded to 6 dp)
    "output": JsonScalar,  # Computed result
    "unit": str,           # "pct", "ratio", "bool"
}
```

**Traced fields** (11 entries from `trace.py:44-166`):

| Key | Formula String | Unit |
|-----|----------------|------|
| `setup_score` | `"0.6 * quality_score + 0.4 * readiness_score"` | pct |
| `distance_to_pivot_pct` | `"100 * (close - pivot_price) / pivot_price"` | pct |
| `atr14_pct` | `"100 * ATR14 / close"` | pct |
| `atr14_pct_trend` | `"slope(atr14_pct, window=20)"` | pct |
| `bb_width_pct` | `"100 * (bb_upper - bb_lower) / bb_mid"` | pct |
| `bb_width_pctile_252` | `"pct_rank(bb_width_pct, window=252)"` | pct |
| `volume_vs_50d` | `"volume / SMA(volume, 50)"` | ratio |
| `rs` | `"close / benchmark_close"` | ratio |
| `rs_line_new_high` | `"rs_current >= max(rs[-252:])"` | bool |
| `rs_vs_spy_65d` | `"100 * (rs / rs[-65] - 1)"` | pct |
| `rs_vs_spy_trend_20d` | `"slope(rs, window=20)"` | ratio |

Score trace is **opt-in**: enabled by passing `include_score_trace=True` to `build_setup_engine_payload()`. When not opted in, `score_trace` is absent from the explain payload.

---

## 7. Candidate Sub-Object Reference

Each entry in the `candidates` list is a `PatternCandidate` TypedDict (`models.py:43-69`):

| Field | Type | Description |
|-------|------|-------------|
| `pattern` | `str` | Pattern label (e.g., `"vcp"`, `"three_weeks_tight"`) |
| `timeframe` | `Literal["daily","weekly"]` | Data horizon used |
| `source_detector` | `str \| None` | Detector that produced this candidate |
| `pivot_price` | `float \| None` | Detector-specific pivot price |
| `pivot_type` | `str \| None` | Pivot family |
| `pivot_date` | `str \| None` | ISO-8601 date |
| `distance_to_pivot_pct` | `float \| None` | Distance from close to pivot |
| `setup_score` | `float \| None` | **Per-candidate** setup score (different blend from top-level) |
| `quality_score` | `float \| None` | Calibrated quality (0–100) |
| `readiness_score` | `float \| None` | Calibrated readiness (0–100) |
| `confidence` | `float \| None` | Calibrated confidence (0–1 ratio) |
| `confidence_pct` | `float \| None` | `confidence × 100` (derived alias) |
| `metrics` | `dict` | Detector-specific measurements |
| `checks` | `dict[str, bool]` | Detector-specific pass/fail flags |
| `notes` | `list[str]` | Annotations (e.g., `"cross_detector_calibration_v1_applied"`) |

### Per-Candidate Setup Score

The per-candidate `setup_score` uses a **three-factor blend** (different from the top-level two-factor blend):

```python
CANDIDATE_SETUP_SCORE_WEIGHTS = (0.55, 0.35, 0.10)
candidate_setup_score = clamp(
    0.55 × calibrated_quality + 0.35 × calibrated_readiness + 0.10 × (calibrated_confidence × 100),
    0, 100
)
```

**Source:** `calibration.py:176-187`

### Calibration Metrics

After calibration, each candidate's `metrics` dict is enriched with:

| Key | Description |
|-----|-------------|
| `calibration_version` | Always `"cross_detector_v1"` |
| `calibration_source_detector` | Detector name used for profile lookup |
| `raw_quality_score` | Pre-calibration quality |
| `raw_readiness_score` | Pre-calibration readiness |
| `raw_confidence` | Pre-calibration confidence (0–1) |
| `raw_confidence_pct` | Pre-calibration confidence (0–100) |
| `normalized_quality_score_0_1` | Quality mapped to 0–1 via profile |
| `normalized_readiness_score_0_1` | Readiness mapped to 0–1 via profile |
| `normalized_confidence_0_1` | Confidence mapped to 0–1 via profile |
| `calibrated_quality_score` | Post-calibration quality (0–100) |
| `calibrated_readiness_score` | Post-calibration readiness (0–100) |
| `calibrated_confidence` | Post-calibration confidence (0–1) |
| `calibrated_confidence_pct` | Post-calibration confidence (0–100) |
| `aggregation_rank_score` | Composite rank score for primary selection |
| `candidate_setup_score` | Three-factor setup score |
| `setup_score_method` | Always `"candidate_blend_v1"` |

### Normalization Formula

```python
normalized = clamp((raw - profile_min) / (profile_max - profile_min), 0, 1)
calibrated = clamp(normalized × 100, 0, 100)
```

**Source:** `calibration.py:289-294`

**SE-H1 cross-ref:** [§9 Cross-Detector Calibration Profiles](se_h1_parameter_catalog.md#9-cross-detector-calibration-profiles)

---

## 8. Pipeline Walkthrough

A single stock flows through the Setup Engine in 8 phases. This walkthrough shows which fields are produced at each stage and the data dependencies between them.

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Input: OHLCV price_data (daily), benchmark_data (SPY)     │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─ Phase 1: Data Policy ──────────────────────────────────────┐
  │  evaluate_setup_engine_data_policy()                        │
  │  Counts: daily_bars, weekly_bars, benchmark_bars            │
  │  Produces: data_policy_result (ok | degraded | insufficient)│
  │  If insufficient → all fields nulled, skip to Phase 7       │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─ Phase 2: Detectors (7) ────────────────────────────────────┐
  │  SetupEngineAggregator.aggregate()                          │
  │  Runs: vcp, three_weeks_tight, high_tight_flag,             │
  │        cup_with_handle, first_pullback, nr7_inside_day,     │
  │        double_bottom (stub)                                 │
  │  Produces: candidates[] (raw), passed_checks, failed_checks │
  │            pattern_primary, pivot_price/type/date            │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─ Phase 3: Calibration ──────────────────────────────────────┐
  │  calibrate_candidates_for_aggregation()                     │
  │  Normalizes raw → calibrated scores per detector profile    │
  │  Produces: calibrated quality_score, readiness_score,       │
  │            confidence, aggregation_rank_score                │
  │  Primary selection: max(rank_score) + structural tie-break  │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─ Phase 4: Readiness Features ───────────────────────────────┐
  │  compute_breakout_readiness_features()                      │
  │  Produces: atr14_pct, atr14_pct_trend, bb_width_pct,       │
  │            bb_width_pctile_252, volume_vs_50d, rs,          │
  │            rs_line_new_high, rs_vs_spy_65d,                 │
  │            rs_vs_spy_trend_20d, distance_to_pivot_pct       │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─ Phase 5: Context ──────────────────────────────────────────┐
  │  _compute_context()                                         │
  │  Produces: stage (1-4), ma_alignment_score, rs_rating       │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─ Phase 6: Score Synthesis ──────────────────────────────────┐
  │  setup_score = 0.60 × quality_score + 0.40 × readiness     │
  │  (Only when both quality and readiness are non-None)        │
  │  Produces: setup_score                                      │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─ Phase 7: Gate Evaluation ──────────────────────────────────┐
  │  build_explain_payload()                                    │
  │  10 gates evaluated sequentially                            │
  │  Produces: explain.passed_checks, explain.failed_checks     │
  │            → setup_ready (derived: zero failures = True)    │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─ Phase 8: Operational Flags ────────────────────────────────┐
  │  compute_operational_flags()                                │
  │  Checks: too_extended, breaks_50d_support, low_liquidity,   │
  │          earnings_soon                                      │
  │  Produces: explain.invalidation_flags                       │
  │  Note: flags do NOT affect setup_ready                      │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  ┌─ Output ────────────────────────────────────────────────────┐
  │  SetupEnginePayload persisted in details_json.setup_engine  │
  │  ScreenerResult.score = setup_score (or 0.0 if None)       │
  │  ScreenerResult.passes = setup_ready                       │
  │  ScreenerResult.rating = calculate_rating(score, details)  │
  └─────────────────────────────────────────────────────────────┘
```

### Tracing "Why is field X this value?"

To diagnose a field value, trace upstream through the pipeline:

| Field | Produced in Phase | Depends on |
|-------|-------------------|------------|
| `setup_ready` | 7 (Gate Evaluation) | All 10 gate inputs + pre-existing checks |
| `setup_score` | 6 (Score Synthesis) | `quality_score`, `readiness_score` |
| `quality_score` | 3 (Calibration) | Primary candidate's raw quality + detector profile |
| `readiness_score` | 3 (Calibration) | Primary candidate's raw readiness + detector profile |
| `pattern_primary` | 2→3 (Detection + Selection) | All candidates, rank_score, confidence floor |
| `distance_to_pivot_pct` | 4 (Readiness) | `close`, `pivot_price` |
| `atr14_pct` | 4 (Readiness) | OHLCV bars (14-period ATR) |
| `stage` | 5 (Context) | MA(50), MA(150), MA(200) values |
| `rs_rating` | 5 (Context) | SPY benchmark close series |
| `too_extended` flag | 8 (Operational) | `distance_to_pivot_pct` |

---

# Part B — Operator Runbook

## 9. Quick Triage Checklist

60-second first-response checklist when something looks wrong with Setup Engine results:

```
□ Is SE enabled?      → Check SETUP_ENGINE_ENABLED env var
                        (False = SE excluded from composite scores)

□ Is Redis running?   → python scripts/inspect_redis.py
                        (SE needs price cache for OHLCV history)

□ Is SPY data fresh?  → Check benchmark cache (< 24h old)
                        redis-cli -n 2 KEYS '*SPY*' | head
                        (Stale SPY → degraded RS fields)

□ Are gates green?    → make gates
                        (All 5 quality gates should pass)

□ Scans completing?   → Check Celery logs for scan task completion
                        tail -f backend/celery_worker.log | grep "SE timing"
                        (Look for errors/timeouts)

□ Database correct?   → Verify DATABASE_URL points to data/stockscanner.db
                        (Wrong path = empty results)
```

**Cross-reference:** [SE-H2 §8 Emergency & Rollback Procedures](se_h2_staged_rollout_plan.md#8-emergency--rollback-procedures) for rollback instructions.

---

## 10. Threshold Tuning Playbook

### Side-Effect Table

Before adjusting any parameter, review its side effects:

| Parameter | What Changes | Side Effects | Validation Steps |
|-----------|-------------|--------------|------------------|
| `readiness_score_ready_min_pct` | Gate 3 threshold | Directly controls setup_ready throughput; most impactful single knob | `make gate-1 && make gate-5` |
| `quality_score_min_pct` | Gate 2 threshold | Must remain ≤ `readiness_score_ready_min_pct` (guardrail) | `make gates` |
| `setup_score_min_pct` | Gate 1 threshold | Downstream of quality/readiness synthesis; raising this may have no effect if Gates 2/3 already filter | `make gate-1` |
| `early_zone_distance_to_pivot_pct_min/max` | Gate 4 window | Wider window = more stocks in buy zone but higher chase risk; min must ≤ max | `make gate-1 && make gate-5` |
| `atr14_pct_max_for_ready` | Gate 5 cap | Higher = accept volatile leaders; lower = filter disorderly setups | `make gate-5` |
| `volume_vs_50d_min_for_ready` | Gate 6 floor | Higher = require volume confirmation; lower = accept quiet accumulation | `make gate-5` |
| `context_ma_alignment_min_pct` | Gate 9 threshold | Higher = stronger uptrend required; None passes (permissive) | `make gates` |
| `context_rs_rating_min` | Gate 10 threshold | Higher = stronger RS required; None passes (permissive) | `make gates` |
| `too_extended_pivot_distance_pct` | Operational flag only | Does NOT affect setup_ready; informational warning | No gate impact |
| `breaks_50d_support_cushion_pct` | Operational flag only | Does NOT affect setup_ready; hard flag but informational | No gate impact |
| `low_liquidity_adtv_min_usd` | Operational flag only | Does NOT affect setup_ready | No gate impact |
| `earnings_soon_window_days` | Operational flag only | Does NOT affect setup_ready | No gate impact |

### Multi-Parameter Change Protocol

When adjusting multiple parameters simultaneously:

1. **Adjust** the parameter(s) via `build_setup_engine_parameters(overrides={...})`
2. **Run** `make gate-1` to verify detector correctness with new values
3. **Run** `make gate-5` to verify golden regression snapshots
4. **If golden snapshots changed**, review the diff and regenerate: `make golden-update`
5. **Run** `make gates` for full validation
6. **Document** the change rationale

**SE-H1 cross-ref:** [§6 Recalibration Guide](se_h1_parameter_catalog.md#6-recalibration-guide) for recommended tuning sequences and market-regime adjustments.

---

## 11. Common Failure Modes

| Failure Mode | Symptom | Root Cause | Detection | Remediation |
|-------------|---------|------------|-----------|-------------|
| Missing benchmark | All RS fields `None`, `data_policy:degraded` in flags | SPY cache expired or Redis down | Check `explain.invalidation_flags` for `missing_or_short_benchmark_history` | `python scripts/inspect_redis.py`; restart Celery data_fetch worker |
| Sparse history | `setup_score = None`, `insufficient_data` check | Stock has < 252 daily bars | Check `explain.failed_checks` for `insufficient_daily_bars` | Expected for IPOs/new listings — no action needed |
| Outlier volume | `volume_vs_50d` extremely high/low | Corporate action (split, dividend) skewing volume SMA | Manual inspection of volume series | Data self-heals after 50 sessions; or clear Redis price cache |
| Stale cache | Old prices producing stale scores | Redis price cache not refreshed | Compare `pivot_date` to current date; check cache TTL | `python scripts/clear_redis_price_cache.py` |
| All stocks not-ready | Zero `setup_ready = True` results | Threshold too tight for current market conditions | Count ready stocks: `SELECT COUNT(*) FROM ... WHERE ... setup_ready = true` | Review [SE-H1 §6 Recalibration Guide](se_h1_parameter_catalog.md#6-recalibration-guide) |
| Detector errors | `{detector}:error` in failed_checks | Exception in detector code | Check Celery logs for `SetupEngineScanner error` | Fix detector bug; errored detector is skipped, others still run |
| Incomplete week | `current_week_incomplete_exclude_from_weekly` in flags | Running mid-week | Check `data_policy_result.requires_weekly_exclude_current` | Expected behavior — weekly features exclude partial week to avoid look-ahead bias |
| SQLite locks | Backfill script hangs or fails | Concurrent write contention | Check for `OperationalError: database is locked` in logs | Run backfill during low-activity; use `--symbols` for smaller batches |

---

## 12. Operational Troubleshooting Decision Trees

### Tree 1: "Stock expected ready but showing not-ready"

```
1. Query the failed checks:
   SQL: SELECT json_extract(details_json, '$.setup_engine.explain.failed_checks')
        FROM stock_feature_daily
        WHERE symbol = '{SYMBOL}'
        ORDER BY date DESC LIMIT 1;

2. Identify which gate failed:
   → "setup_score_below_threshold"    → Gate 1: check setup_score vs 65.0
   → "quality_below_threshold"        → Gate 2: check quality_score vs 60.0
   → "readiness_below_threshold"      → Gate 3: check readiness_score vs 70.0
   → "outside_early_zone"             → Gate 4: check distance_to_pivot_pct vs [-2, 3]
   → "atr14_pct_exceeds_limit"        → Gate 5: check atr14_pct vs 8.0
   → "volume_below_minimum"           → Gate 6: check volume_vs_50d vs 1.0
   → "rs_leadership_insufficient"     → Gate 7: check rs_vs_spy_65d and rs_line_new_high
   → "stage_not_ok"                   → Gate 8: check stage value (must be 1 or 2)
   → "ma_alignment_insufficient"      → Gate 9: check ma_alignment_score vs 60.0
   → "rs_rating_insufficient"         → Gate 10: check rs_rating vs 50.0
   → "insufficient_data"              → Data policy: check bar counts
   → "no_primary_pattern"             → No detectors found a pattern

3. Compare the failing value against the threshold:
   SQL: SELECT json_extract(details_json, '$.setup_engine.{field_name}')
        FROM stock_feature_daily
        WHERE symbol = '{SYMBOL}'
        ORDER BY date DESC LIMIT 1;

4. Decide: is the value genuinely wrong, or is the threshold too tight?
   → Value wrong:    trace upstream (see Pipeline Walkthrough §8)
   → Threshold tight: see SE-H1 §6 Recalibration Guide
```

### Tree 2: "Stock has no setup_engine data"

```
1. Check if SE data exists at all:
   SQL: SELECT COUNT(*)
        FROM stock_feature_daily
        WHERE symbol = '{SYMBOL}'
          AND details_json LIKE '%setup_engine%';

2. If count = 0:
   → Was the stock in the scan universe?
     SQL: SELECT * FROM stock_universe WHERE symbol = '{SYMBOL}';
   → Was SE enabled during the scan?
     Check SETUP_ENGINE_ENABLED env var
   → Was the scan completed?
     Check Celery logs for the symbol

3. If count > 0 but latest row missing SE:
   → Check if the most recent scan included SE:
     SQL: SELECT date, json_extract(details_json, '$.setup_engine.schema_version')
          FROM stock_feature_daily
          WHERE symbol = '{SYMBOL}'
          ORDER BY date DESC LIMIT 5;
   → If older rows have SE but newer don't: scan may have been run
     with SE disabled after being previously enabled

4. To backfill missing data:
   python scripts/backfill_setup_engine.py --symbols {SYMBOL} --dry-run
   python scripts/backfill_setup_engine.py --symbols {SYMBOL} --yes
```

### Tree 3: "Setup score seems wrong"

```
1. Check the score trace (if available):
   SQL: SELECT json_extract(details_json, '$.setup_engine.explain.score_trace')
        FROM stock_feature_daily
        WHERE symbol = '{SYMBOL}'
        ORDER BY date DESC LIMIT 1;

2. Verify quality/readiness inputs:
   SQL: SELECT json_extract(details_json, '$.setup_engine.quality_score') as quality,
               json_extract(details_json, '$.setup_engine.readiness_score') as readiness,
               json_extract(details_json, '$.setup_engine.setup_score') as setup
        FROM stock_feature_daily
        WHERE symbol = '{SYMBOL}'
        ORDER BY date DESC LIMIT 1;

   Expected: setup_score ≈ 0.60 × quality + 0.40 × readiness

3. If quality/readiness seem wrong, check calibration:
   SQL: SELECT json_extract(details_json, '$.setup_engine.candidates')
        FROM stock_feature_daily
        WHERE symbol = '{SYMBOL}'
        ORDER BY date DESC LIMIT 1;

   Look at candidate metrics for:
   - raw_quality_score vs calibrated_quality_score
   - raw_readiness_score vs calibrated_readiness_score
   - calibration_source_detector (correct profile?)

4. If raw scores are wrong:
   → Problem is in the detector, not calibration
   → Check detector-specific logic for the pattern type
```

### Tree 4: "Backfill script failures"

```
1. Check SPY benchmark availability:
   python -c "
   from app.database import SessionLocal
   from sqlalchemy import text
   s = SessionLocal()
   r = s.execute(text(\"SELECT COUNT(*) FROM stock_prices WHERE symbol='SPY'\"))
   print(f'SPY bars: {r.scalar()}')
   s.close()
   "

2. Check price cache:
   python scripts/inspect_redis.py
   (Look for price cache keys, check TTLs)

3. If SQLite lock errors:
   → Ensure no Celery workers are running scans
   → Use --symbols for smaller batches
   → Run during low-activity periods

4. Preview before committing:
   python scripts/backfill_setup_engine.py --dry-run

5. For specific date ranges:
   python scripts/backfill_setup_engine.py --date-from 2025-01-01 --date-to 2025-06-30 --yes
```

### Tree 5: "Performance degradation"

```
1. Check per-detector timing:
   → Enable DEBUG logging and look for:
     "SE detector {name}: outcome={} candidates={} elapsed={:.1f}ms"
   → Or run: make gate-4

2. If a specific detector is slow:
   → Check computational caps (e.g., cup_handle _MAX_SWING_PAIR_EVALUATIONS=3000)
   → Check input data size (daily_bars, weekly_bars)

3. If overall pipeline is slow:
   → Check total timing in INFO logs:
     "SE timing {symbol}: prep={:.1f}ms detectors={:.1f}ms readiness={:.1f}ms total={:.1f}ms"
   → Normal: < 200ms per stock
   → Degraded: 200-500ms (investigate detector count)
   → Critical: > 500ms (possible data issue or runaway computation)

4. Run performance gate:
   make gate-4
   (Advisory gate — reports timing but does not block)
```

---

## 13. Ownership & Escalation Map

| Issue Category | First Responder | Escalation Path | When to Escalate |
|---------------|-----------------|-----------------|------------------|
| Scanner/scoring bugs | Backend developer | Setup Engine maintainer | Gate failures persist after fix attempt |
| Data quality issues | Operator | Backend developer | Stale cache persists after `clear_redis_price_cache.py` |
| Frontend display | Frontend developer | Backend developer | If display issue is actually a data issue |
| Performance degradation | Operator | Backend developer | `make gate-4` shows > 500ms per stock consistently |
| Infrastructure (Redis, Celery) | Operator | System administrator | Service won't restart or data loss suspected |
| Parameter tuning | Operator (self-service) | Setup Engine maintainer | Guardrail violations or unexpected throughput changes |
| Backfill failures | Operator | Backend developer | Script fails on valid data |

### Diagnostic Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/inspect_redis.py` | Inspect Redis cache keys and TTLs | `python scripts/inspect_redis.py` |
| `scripts/cache_diagnostic.py` | Trace cache flow (DB → Redis) | `python scripts/cache_diagnostic.py` |
| `scripts/check_cache_status.py` | Check price cache status | `python scripts/check_cache_status.py` |
| `scripts/clear_redis_price_cache.py` | Clear Redis cache after config change | `python scripts/clear_redis_price_cache.py` |
| `scripts/force_full_cache_refresh.py` | Force full cache refresh | `python scripts/force_full_cache_refresh.py` |
| `scripts/cleanup_orphaned_scans.py` | Clean up stale scans, VACUUM DB | `python scripts/cleanup_orphaned_scans.py` |
| `scripts/backfill_setup_engine.py` | Backfill SE payloads into existing rows | `python scripts/backfill_setup_engine.py --help` |

---

## 14. Keeping This Document Current

### Update Triggers

| Event | What to Update |
|-------|---------------|
| New detector added | §1 (field index if new fields), §3 (pivot derivation table), §6 (check strings), §8 (pipeline diagram) |
| New readiness gate added | §1, §6 (check string table), §10 (side-effect table), §12 (decision tree 1) |
| Gate logic changed | §6 (check strings), §10 (side-effect table) |
| Parameter added/modified | §10 (side-effect table) — threshold values live in SE-H1, not here |
| New operational flag | §6 (flag codes table) |
| Score synthesis formula changed | §2 (`setup_score`), §7 (candidate setup_score), §8 (phase 6) |
| New readiness feature | §1 (field index), §4 (feature section), §8 (phase 4) |
| New diagnostic script | §13 (scripts table) |

### Source-of-Truth Mapping

| Document Content | Canonical Source |
|-----------------|-----------------|
| Field specs (names, types, units) | `SETUP_ENGINE_FIELD_SPECS` in `models.py` |
| Readiness formulas | `_compute_readiness_core()` in `readiness.py` |
| Gate evaluation logic | `build_explain_payload()` in `explain_builder.py` |
| Operational flag conditions | `compute_operational_flags()` in `operational_flags.py` |
| Score trace formulas | `build_score_trace()` in `trace.py` |
| Calibration profiles | `_PROFILE_BY_DETECTOR` in `calibration.py` |
| Primary selection algorithm | `_select_primary_candidate()` in `aggregator.py` |
| Context computation | `_compute_context()` in `setup_engine_screener.py` |
| Score synthesis weights | `SETUP_SCORE_WEIGHTS` in `config.py` |
| Rating rules | `calculate_rating()` in `setup_engine_screener.py` |
| Parameter values and bounds | `SETUP_ENGINE_PARAMETER_SPECS` in `config.py` |
| Data policy requirements | `SetupEngineDataRequirements` in `policy.py` |

### Cross-Reference Pattern

This document follows the SE-H1/SE-H2 convention: threshold **values** and **bounds** are owned by [SE-H1](se_h1_parameter_catalog.md) and cross-referenced here rather than duplicated. When in doubt about a threshold value, SE-H1 is the source of truth; this document explains what the field *means* and how it *behaves*.

**SE-H1 cross-ref:** [§12 Keeping This Catalog Current](se_h1_parameter_catalog.md#12-keeping-this-catalog-current) for the parallel update protocol.
