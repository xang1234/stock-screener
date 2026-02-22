# SE-H1: Parameter Catalog and Calibration Report

> **Definitive reference** for every configurable threshold, compile-time constant, calibration profile, and readiness gate in the Setup Engine. Intended for operators tuning runtime behavior and developers modifying detection logic.

## Table of Contents

1. [Quick-Reference Tuning Card](#1-quick-reference-tuning-card)
2. [Score Synthesis Architecture](#2-score-synthesis-architecture)
3. [Setup Ready Gate Map (10 Gates)](#3-setup-ready-gate-map-10-gates)
4. [Configurable Threshold Catalog (Runtime-Tunable)](#4-configurable-threshold-catalog-runtime-tunable)
5. [Parameter Interactions](#5-parameter-interactions)
6. [Recalibration Guide](#6-recalibration-guide)
7. [Worked Example: Stock Flowing Through the Pipeline](#7-worked-example-stock-flowing-through-the-pipeline)
8. [Calibration Methodology](#8-calibration-methodology)
9. [Cross-Detector Calibration Profiles](#9-cross-detector-calibration-profiles)
10. [Detector-Specific Constants (Compile-Time)](#10-detector-specific-constants-compile-time)
11. [Data Requirements & Feature Computation](#11-data-requirements--feature-computation)
12. [Keeping This Catalog Current](#12-keeping-this-catalog-current)

### Related Design Docs

- [SE-A3: Parameter Governance](se_a3_parameter_governance.md) — Rationale for the parameter dataclass and validation architecture
- [SE-A4: Data Requirements Policy](se_a4_data_requirements_policy.md) — Detailed data sufficiency semantics
- [SE-C7: Cross-Detector Calibration](se_c7_cross_detector_calibration.md) — Normalization algebra and rank-score derivation

### Canonical Source Files

| File | Role |
|------|------|
| `backend/app/analysis/patterns/config.py` | Parameter dataclass, specs, validation, weights |
| `backend/app/analysis/patterns/calibration.py` | Detector calibration profiles |
| `backend/app/analysis/patterns/policy.py` | Data requirements |
| `backend/app/analysis/patterns/readiness.py` | Readiness feature computation |
| `backend/app/analysis/patterns/explain_builder.py` | 10-gate evaluation |
| `backend/app/analysis/patterns/operational_flags.py` | Invalidation flags |
| `backend/app/scanners/setup_engine_screener.py` | Rating rules |
| `backend/app/scanners/setup_engine_scanner.py` | Payload assembly, score synthesis |
| `backend/app/analysis/patterns/detectors/` | Detector constants (7 files) |

---

## 1. Quick-Reference Tuning Card

The six most commonly adjusted parameters and their directional effects:

| Knob | Default | Turn ↑ | Turn ↓ |
|------|---------|--------|--------|
| `readiness_score_ready_min_pct` | 70.0 | Fewer, higher-conviction signals | More setups marked ready |
| `quality_score_min_pct` | 60.0 | Reject weak bases earlier | More pattern diversity |
| `setup_score_min_pct` | 65.0 | Tighter composite gate | More throughput |
| `early_zone_distance_to_pivot_pct_max` | 3.0 | Allow later entries | Tighter buy-zone window |
| `atr14_pct_max_for_ready` | 8.0 | Accept volatile leaders | Filter disorderly setups |
| `context_rs_rating_min` | 50.0 | Stronger RS filter | Catch early-stage movers |

> **Warning**: These parameters interact — see [Parameter Interactions](#5-parameter-interactions) before tuning multiple knobs simultaneously.

Source: `config.py:34-71`

---

## 2. Score Synthesis Architecture

### Weight Systems

Two weight tuples govern score composition at different pipeline stages:

**Top-level setup score** (used in payload assembly and Gate 1):
```
SETUP_SCORE_WEIGHTS = (0.60 quality, 0.40 readiness)
setup_score = 0.60 × quality_score + 0.40 × readiness_score
```
Source: `config.py:18`

**Per-candidate setup score** (used during cross-detector comparison):
```
CANDIDATE_SETUP_SCORE_WEIGHTS = (0.55 quality, 0.35 readiness, 0.10 confidence)
candidate_setup_score = 0.55 × quality + 0.35 × readiness + 0.10 × (confidence × 100)
```
Source: `config.py:16`

### Rating Rules

After setup_score and setup_ready are computed, the screener assigns a human-readable rating:

| Condition | Rating |
|-----------|--------|
| score ≥ 80 **and** setup_ready = True | **Strong Buy** |
| score ≥ 65 **and** setup_ready = True | **Buy** |
| score ≥ 50 | **Watch** |
| Otherwise | **Pass** |

Source: `setup_engine_screener.py:89-98`

---

## 3. Setup Ready Gate Map (10 Gates)

The explain builder evaluates 10 sequential gates. `setup_ready` is True only when **all** gates pass (zero failed checks from any source).

| Gate | Name | Check Label | Threshold Parameter | Default | Permissive? | Passes When | Behavioral Note |
|------|------|-------------|--------------------|---------|----|-------------|-----------------|
| 1 | Setup Score | `setup_score_ok` | `setup_score_min_pct` | 65.0 | No | `setup_score ≥ 65.0` | Composite of quality + readiness |
| 2 | Quality Floor | `quality_floor_ok` | `quality_score_min_pct` | 60.0 | No | `quality_score ≥ 60.0` | Rejects weak bases before composite |
| 3 | Readiness Floor | `readiness_floor_ok` | `readiness_score_ready_min_pct` | 70.0 | No | `readiness_score ≥ 70.0` | Primary readiness threshold |
| 4 | Early Zone | `in_early_zone` | `early_zone_*_min/max` | [-2.0, 3.0] | **No** (None → fail) | `min ≤ distance_to_pivot ≤ max` | Only non-permissive feature gate |
| 5 | ATR14 Cap | `atr14_within_limit` | `atr14_pct_max_for_ready` | 8.0 | **Yes** (None → pass) | `atr14_pct ≤ 8.0` | Filters disorderly volatility |
| 6 | Volume Floor | `volume_sufficient` | `volume_vs_50d_min_for_ready` | 1.0 | **Yes** (None → pass) | `volume_vs_50d ≥ 1.0` | Minimum liquidity participation |
| 7 | RS Leadership | `rs_leadership_ok` | (implicit > 0) | N/A | **Yes** (both None → pass) | `rs_vs_spy_65d > 0` or `rs_line_new_high` | Confirms relative outperformance |
| 8 | Stage | `stage_ok` | (implicit 1 or 2) | N/A | **Semi** (None → pass) | `stage ∈ {1, 2}` or `None` | Stage 3/4 explicitly fail |
| 9 | MA Alignment | `ma_alignment_ok` | `context_ma_alignment_min_pct` | 60.0 | **Yes** (None → pass) | `ma_alignment_score ≥ 60.0` | Minervini MA stack (50>150>200) |
| 10 | RS Rating | `rs_rating_ok` | `context_rs_rating_min` | 50.0 | **Yes** (None → pass) | `rs_rating ≥ 50.0` | Linear scale; 50 = outperforming SPY |

**Permissive gate stacking**: Gates 5–10 are permissive (None passes). A stock with missing RS data passes Gates 7, 9, and 10 automatically. A stock with no benchmark data at all passes all six permissive gates. This means `setup_ready` depends only on Gates 1–4 for sparse-data stocks. This is a **deliberate design choice** — newly-listed stocks with sparse data aren't automatically excluded; they are evaluated on the data that exists.

Source: `explain_builder.py:71-182`

---

## 4. Configurable Threshold Catalog (Runtime-Tunable)

These 18 parameters are tunable at runtime via `build_setup_engine_parameters(overrides={...})` — no code changes needed. Unknown override keys are rejected with a `KeyError`.

### 4.1 Pattern Contraction Thresholds (3WT)

| Parameter | Default | Min | Max | Unit | Profile | Rationale | Effect of Raising |
|-----------|---------|-----|-----|------|---------|-----------|-------------------|
| `three_weeks_tight_max_contraction_pct_strict` | 1.0 | 0.2 | 5.0 | pct | strict | Strict 3WT mode should only allow very tight contractions | More patterns qualify as "strict" (looser definition of tight) |
| `three_weeks_tight_max_contraction_pct_relaxed` | 1.5 | 0.5 | 8.0 | pct | relaxed | Relaxed 3WT mode supports noisy leaders while preserving shape quality | More patterns detected in relaxed mode |

**Guardrail**: `strict ≤ relaxed`. Violating this produces a validation error.

### 4.2 Breakout Zone Window

| Parameter | Default | Min | Max | Unit | Profile | Rationale | Effect of Raising |
|-----------|---------|-----|-----|------|---------|-----------|-------------------|
| `early_zone_distance_to_pivot_pct_min` | -2.0 | -15.0 | 5.0 | pct | baseline | Allows pre-breakout setups slightly below pivot without deep failure states | Requires price closer to or above pivot |
| `early_zone_distance_to_pivot_pct_max` | 3.0 | -5.0 | 20.0 | pct | baseline | Prevents late-chase entries from being labeled as early-zone ready | Widens the buy zone above pivot |

**Guardrail**: `min ≤ max`. The default window [-2%, +3%] means a stock is "in the early zone" when it is between 2% below pivot and 3% above.

### 4.3 Volatility Squeeze (Bollinger Width Percentile)

| Parameter | Default | Min | Max | Unit | Profile | Rationale | Effect of Raising |
|-----------|---------|-----|-----|------|---------|-----------|-------------------|
| `squeeze_bb_width_pctile_max_strict` | 20.0 | 1.0 | 50.0 | pct | strict | Strict squeeze requires historically compressed volatility | More observations qualify as "squeezed" |
| `squeeze_bb_width_pctile_max_relaxed` | 35.0 | 5.0 | 80.0 | pct | relaxed | Relaxed squeeze tolerates broader consolidations in liquid names | Less restrictive on volatility compression |

**Guardrail**: `strict ≤ relaxed`.

### 4.4 Score Gates

| Parameter | Default | Min | Max | Unit | Profile | Rationale | Effect of Raising |
|-----------|---------|-----|-----|------|---------|-----------|-------------------|
| `readiness_score_ready_min_pct` | 70.0 | 40.0 | 95.0 | pct | baseline | 70% balances signal quality and setup throughput for v1 rollout | Fewer setups pass Gate 3 |
| `quality_score_min_pct` | 60.0 | 30.0 | 90.0 | pct | baseline | Avoids classifying low-quality bases as tradeable setups | Fewer setups pass Gate 2 |
| `pattern_confidence_min_pct` | 55.0 | 20.0 | 95.0 | pct | baseline | Ensures primary pattern selection is not dominated by weak detector noise | Only high-confidence patterns qualify |
| `setup_score_min_pct` | 65.0 | 30.0 | 95.0 | pct | baseline | Minimum composite setup score combining quality and readiness | Fewer setups pass Gate 1 |

**Guardrail**: `quality_score_min_pct ≤ readiness_score_ready_min_pct`. Readiness is always the tighter filter. Attempting to set quality=75, readiness=70 produces a validation error.

### 4.5 Supplemental Readiness Controls

| Parameter | Default | Min | Max | Unit | Profile | Rationale | Effect of Raising |
|-----------|---------|-----|-----|------|---------|-----------|-------------------|
| `atr14_pct_max_for_ready` | 8.0 | 1.0 | 30.0 | pct | baseline | Caps volatility to avoid breakout-ready flags in disorderly conditions | Accepts more volatile names |
| `volume_vs_50d_min_for_ready` | 1.0 | 0.2 | 5.0 | ratio | baseline | Requires at least baseline liquidity participation at decision time | Higher bar for volume confirmation |

### 4.6 Context Gates

| Parameter | Default | Min | Max | Unit | Profile | Rationale | Effect of Raising |
|-----------|---------|-----|-----|------|---------|-----------|-------------------|
| `context_ma_alignment_min_pct` | 60.0 | 0.0 | 100.0 | pct | baseline | Minimum Minervini MA alignment score (50>150>200) for Gate 9. Permissive: None passes | Requires stronger uptrend structure |
| `context_rs_rating_min` | 50.0 | 0.0 | 100.0 | pct | baseline | Minimum RS rating for Gate 10. Linear scale: 50 = outperforming SPY. Permissive: None passes | Requires stronger relative strength |

### 4.7 Operational Flags (Informational)

These parameters control **operational invalidation flags**, which are warnings that do **not** affect `setup_ready` or `derived_ready`. They surface risk information for operators.

| Parameter | Default | Min | Max | Unit | Profile | Rationale | Effect of Raising |
|-----------|---------|-----|-----|------|---------|-----------|-------------------|
| `too_extended_pivot_distance_pct` | 10.0 | 2.0 | 50.0 | pct | baseline | O'Neil: buying >5–8% past pivot is chasing. Flags extended entries | Fewer stocks flagged as "too extended" |
| `breaks_50d_support_cushion_pct` | 0.0 | 0.0 | 10.0 | pct | baseline | 0 = strict (any close below 50d MA). Higher = allow N% below before flagging | More tolerance for pullback below 50d MA |
| `low_liquidity_adtv_min_usd` | 1,000,000 | 0 | 100,000,000 | usd | baseline | Minimum ADTV (50-day) for meaningful position sizing | Excludes more thinly-traded stocks |
| `earnings_soon_window_days` | 21.0 | 1.0 | 90.0 | days | baseline | ~3 weeks covers pre-earnings risk window | Wider earnings risk zone |

**Flag severity** (from `operational_flags.py`):
- `too_extended` — soft (is_hard=False)
- `breaks_50d_support` — **hard** (is_hard=True)
- `low_liquidity` — soft (is_hard=False)
- `earnings_soon` — soft (is_hard=False)

Source: `config.py:34-236`, `operational_flags.py:37-91`

---

## 5. Parameter Interactions

### 5.1 Quality → Setup Score Chain

`quality_score_min_pct` (Gate 2) vs `setup_score_min_pct` (Gate 1) create a dependency chain.

**Example**: quality=62, readiness=72 → setup_score = 0.60×62 + 0.40×72 = **66.0** — passes Gate 1 (≥ 65). But raising `quality_score_min_pct` from 60 to 65 causes Gate 2 to reject this stock before Gate 1 is even relevant.

### 5.2 Guardrail Constraint: quality ≤ readiness

The validator enforces `quality_score_min_pct ≤ readiness_score_ready_min_pct`. This ensures readiness is always the tighter filter. Attempting to set quality=75, readiness=70 triggers:
```
ValueError: quality_score_min_pct must be <= readiness_score_ready_min_pct
```

### 5.3 Early Zone + Too Extended

Gate 4 filters on `distance_to_pivot ∈ [-2%, 3%]` — this **blocks** `setup_ready`. The `too_extended_pivot_distance_pct=10%` threshold is an operational flag (info-only). A stock at +5% past pivot **fails** Gate 4 (`setup_ready=False`) but would also receive the `too_extended` flag if it were >10%. A stock at +8% fails Gate 4 (not in early zone) but is **not** flagged as too_extended (below 10% threshold).

### 5.4 Permissive Gate Stacking

Gates 5–10 all pass when their input data is `None`. A stock with no benchmark data passes RS gates (7, 10) and MA alignment (9) automatically. Combined with missing volume and ATR data, such a stock's `setup_ready` depends entirely on Gates 1–4. This means sparse-data stocks can achieve `setup_ready=True` if their quality/readiness scores and pivot distance are sufficient.

### 5.5 3WT Strict/Relaxed vs Squeeze Strict/Relaxed

Both the 3WT contraction thresholds and the squeeze BB-width thresholds have dual strict/relaxed profiles, but they serve entirely different detector families. Changing `three_weeks_tight_max_contraction_pct_strict` has **zero** effect on squeeze detection, and vice versa. They are independent parameter pairs.

---

## 6. Recalibration Guide

### Increasing Throughput (More Setups Marked Ready)

1. Lower `readiness_score_ready_min_pct` (e.g., 70 → 60)
2. Widen `early_zone_distance_to_pivot_pct_max` (e.g., 3.0 → 5.0)
3. Raise `atr14_pct_max_for_ready` (e.g., 8.0 → 12.0)
4. Lower `quality_score_min_pct` (e.g., 60 → 50)

### Increasing Selectivity (Fewer, Higher-Quality Signals)

1. Raise `readiness_score_ready_min_pct` (e.g., 70 → 80)
2. Raise `quality_score_min_pct` (e.g., 60 → 70)
3. Raise `setup_score_min_pct` (e.g., 65 → 75)
4. Narrow `early_zone_distance_to_pivot_pct_max` (e.g., 3.0 → 2.0)

### Adjusting for Market Regime

- **High-volatility markets**: Raise `atr14_pct_max_for_ready` to avoid filtering out legitimate leaders with elevated ATR. Consider relaxing 3WT contraction thresholds.
- **Low-volume markets**: Lower `volume_vs_50d_min_for_ready` (e.g., 1.0 → 0.7) to avoid penalizing setups during low-participation periods.
- **Narrow leadership**: Lower `context_rs_rating_min` to catch early-stage movers before RS inflection.

### Adding a New Detector

1. Implement the detector module extending `PatternDetector` base class
2. Create a `DetectorCalibrationProfile` with empirical quality/readiness/confidence envelopes
3. Add the profile to `_PROFILE_BY_DETECTOR` in `calibration.py`
4. Add its constants to [Section 10](#10-detector-specific-constants-compile-time)
5. Create its calibration profile entry in [Section 9](#9-cross-detector-calibration-profiles)
6. Update this catalog

### Validation Checklist

After any parameter change:
1. Run `make gates` to verify all quality gates pass
2. If golden snapshots change, regenerate via `make golden-update`
3. Verify the change doesn't violate any of the 4 cross-parameter guardrails:
   - `strict ≤ relaxed` for 3WT contraction
   - `early_zone min ≤ max`
   - `strict ≤ relaxed` for squeeze BB width
   - `quality_score_min_pct ≤ readiness_score_ready_min_pct`

### Guardrail Violations

The four cross-parameter validators in `validate_setup_engine_parameters()` enforce:

| Guardrail | Condition | Error Message |
|-----------|-----------|---------------|
| 3WT contraction | strict > relaxed | `three_weeks_tight_max_contraction_pct_strict must be <= relaxed` |
| Early zone | min > max | `early_zone_distance_to_pivot_pct_min must be <= ...max` |
| Squeeze width | strict > relaxed | `squeeze_bb_width_pctile_max_strict must be <= relaxed` |
| Quality ≤ Readiness | quality > readiness | `quality_score_min_pct must be <= readiness_score_ready_min_pct` |

Additionally, per-parameter bounds are enforced: each parameter must fall within its `[min_value, max_value]` range as defined in `SETUP_ENGINE_PARAMETER_SPECS`. Unknown keys passed to `build_setup_engine_parameters()` raise `KeyError`.

Source: `config.py:241-315`

---

## 7. Worked Example: Stock Flowing Through the Pipeline

### Scenario: "ACME Corp" Setup Evaluation

**Inputs** (after calibration):
- quality_score = 72
- readiness_score = 68
- confidence = 0.78
- distance_to_pivot_pct = +1.5%
- atr14_pct = 5.2%
- volume_vs_50d = 1.3
- rs_vs_spy_65d = +4.2% (positive)
- rs_line_new_high = False
- stage = 2
- ma_alignment_score = 80
- rs_rating = 65

### Setup Score Computation

```
setup_score = 0.60 × 72 + 0.40 × 68
            = 43.2 + 27.2
            = 70.4
```

### Gate Walk-Through

| Gate | Check | Input | Threshold | Result |
|------|-------|-------|-----------|--------|
| 1 | Setup Score | 70.4 | ≥ 65.0 | **PASS** |
| 2 | Quality Floor | 72 | ≥ 60.0 | **PASS** |
| 3 | Readiness Floor | 68 | ≥ 70.0 | **FAIL** |
| 4 | Early Zone | +1.5% | ∈ [-2.0, 3.0] | **PASS** |
| 5 | ATR14 Cap | 5.2% | ≤ 8.0 | **PASS** |
| 6 | Volume Floor | 1.3 | ≥ 1.0 | **PASS** |
| 7 | RS Leadership | +4.2% > 0 | rs_vs_spy > 0 | **PASS** |
| 8 | Stage | 2 | ∈ {1, 2} | **PASS** |
| 9 | MA Alignment | 80 | ≥ 60.0 | **PASS** |
| 10 | RS Rating | 65 | ≥ 50.0 | **PASS** |

**Result**: Gate 3 fails (68 < 70). `setup_ready = False`. Rating = **Watch** (score 70.4 ≥ 50, but not ready).

### Tuning Scenario A: Lower Readiness Threshold

If we lower `readiness_score_ready_min_pct` from 70 → 65:
- Gate 3: 68 ≥ 65 → **PASS**
- All 10 gates pass → `setup_ready = True`
- Rating = **Buy** (score 70.4 ≥ 65, setup_ready=True)

### Tuning Scenario B: Widen Early Zone

If we widen `early_zone_distance_to_pivot_pct_max` from 3.0 → 5.0:
- No change for ACME (already at +1.5%)
- But a stock at +4.5% that previously failed Gate 4 would now pass
- Gate 3 still fails for ACME → `setup_ready = False` regardless

### Tuning Scenario C: Raise Quality Floor

If we raise `quality_score_min_pct` from 60 → 75:
- Gate 2: 72 < 75 → **FAIL**
- ACME now fails at Gate 2 (quality too low) instead of Gate 3

**Note**: Canonical source for formulas is `config.py` (weights) and `explain_builder.py` (gates).

---

## 8. Calibration Methodology

### Literature-Informed Defaults

Threshold values are grounded in William O'Neil's *How to Make Money in Stocks* and Mark Minervini's *Trade Like a Stock Market Wizard* pattern recognition principles:

- **Early zone [-2%, +3%]** reflects O'Neil's "buy within 5% of pivot" rule, tightened for automated screening where positions can be sized more precisely than manual entries.
- **ATR14 cap at 8%** filters the kind of disorderly price action that Minervini describes as "climax top" behavior — high volatility is a feature of late-stage moves, not early breakouts.
- **RS rating minimum of 50** aligns with O'Neil's RS requirement (≥70 for CANSLIM), relaxed here because the Setup Engine targets pre-breakout patterns where RS may still be inflecting upward.

### Design Rationale, Not Backtested Values

The `SetupEngineParameterSpec.rationale` strings reflect **structural reasoning**, not statistical optimization. The golden regression suite validates *consistency* (same inputs → same outputs), not predictive accuracy. Current defaults have **not** been validated against historical win rates. Future calibration work should include forward-testing against known historical setups.

### Strict/Relaxed Dual-Profile

Structural patterns (3WT, squeeze) offer strict/relaxed modes:
- **Strict**: Low-noise environments where institutional accumulation creates genuine tight consolidation (3WT strict: ≤1.0% contraction, squeeze strict: ≤20th percentile BB width)
- **Relaxed**: Volatile leadership where even strong patterns show wider ranges (3WT relaxed: ≤1.5%, squeeze relaxed: ≤35th percentile)

### Calibration Profile Design

Each detector family has a `DetectorCalibrationProfile` that defines expected raw score envelopes. These reflect the structural characteristics of each pattern type:

- **NR7/Inside Day** has compressed envelopes (quality 20–65, confidence 0.20–0.78) because trigger patterns have inherently lower confidence than multi-week structural patterns. A single narrow-range day provides less structural evidence than a 6-week cup formation.
- **High Tight Flag** has the highest confidence bias (+0.03) because HTF patterns are rare, require extreme price appreciation (100%+ pole return), and are high-conviction when found.
- **Cup with Handle** receives moderate positive bias (+0.02) as it represents the most well-studied and validated O'Neil base pattern.

### Confidence Bias System

The `confidence_bias` field in each calibration profile applies a small additive adjustment after normalization:

| Category | Detectors | Bias Range | Rationale |
|----------|-----------|------------|-----------|
| Structural (positive) | VCP (+0.01), HTF (+0.03), CWH (+0.02) | +0.01 to +0.03 | Multi-week patterns with strong structural evidence |
| Neutral | 3WT (0.00) | 0.00 | Already validated by tight consolidation geometry |
| Default (slight negative) | Default profile (-0.01), Double Bottom (-0.01) | -0.01 | Conservative baseline for unspecialized detectors |
| Trigger (negative) | First Pullback (-0.02), NR7/Inside Day (-0.06) | -0.02 to -0.06 | Single-event triggers with inherently lower conviction |

### Known Limitations

- Current defaults have not been validated against historical win rates
- Calibration profiles are based on structural reasoning about pattern quality, not backtested performance
- The confidence bias system assumes structural patterns are inherently higher-conviction than trigger patterns — this may not hold in all market regimes
- Future calibration work should include forward-testing against known historical setups (e.g., VCP examples from Minervini's published trade logs)

---

## 9. Cross-Detector Calibration Profiles

All 8 named profiles plus the default envelope:

| Detector | Quality Range | Readiness Range | Confidence Range | Confidence Bias | Design Intent |
|----------|--------------|-----------------|------------------|----------------|---------------|
| **default** | 35–95 | 35–95 | 0.20–0.95 | -0.01 | Conservative baseline for unknown/new detectors |
| **vcp** | 45–95 | 45–95 | 0.25–0.95 | +0.01 | Legacy VCP with moderate structural evidence |
| **three_weeks_tight** | 45–95 | 50–95 | 0.35–0.95 | 0.00 | Tight consolidation validated by geometry |
| **high_tight_flag** | 45–98 | 60–98 | 0.30–0.95 | **+0.03** | Rare, extreme pattern; highest conviction |
| **cup_with_handle** | 45–95 | 45–95 | 0.30–0.95 | +0.02 | Classic O'Neil base; well-validated |
| **cup_handle** | 45–95 | 45–95 | 0.30–0.95 | +0.02 | Backward-compatibility alias for cup_with_handle |
| **first_pullback** | 40–95 | 35–90 | 0.25–0.95 | -0.02 | Trend-resumption trigger; moderate conviction |
| **nr7_inside_day** | 20–65 | 24–70 | 0.20–0.78 | **-0.06** | Lightweight trigger; compressed scoring envelope |
| **double_bottom** | 45–95 | 45–95 | 0.25–0.95 | -0.01 | Stub detector; matches default-class envelope |

### Structural vs Trigger Distinction

The profiles encode a fundamental distinction between **structural patterns** (multi-week formations with clear geometric validation) and **trigger patterns** (single-day or short-duration signals):

- **Structural patterns** (VCP, 3WT, HTF, CWH) have wide scoring envelopes and positive confidence bias. A raw quality_score of 70 from a cup-with-handle detector carries more conviction than 70 from NR7.
- **Trigger patterns** (First Pullback, NR7/Inside Day) have compressed envelopes and negative confidence bias. Their raw scores are calibrated to a lower ceiling because the underlying evidence is inherently less robust.

The normalization process maps raw detector scores into the profile's `[min, max]` range to produce a 0–1 normalized value, then scales back to 0–100 for the calibrated score. This ensures that identical raw scores from different detectors produce appropriately different calibrated scores.

Source: `calibration.py:29-114`

---

## 10. Detector-Specific Constants (Compile-Time)

These constants are **NOT runtime-configurable** — they are module-level `_PRIVATE` constants requiring code changes to modify. They define the structural geometry of what each detector considers a valid pattern.

### 10.1 Three Weeks Tight

| Constant | Value | Description |
|----------|-------|-------------|
| `_MIN_WEEKS_TIGHT` | 3 | Minimum consecutive tight weeks |
| `_MAX_WEEKS_TIGHT` | 8 | Maximum tight weeks to consider |
| `_MAX_CANDIDATES` | 5 | Maximum candidates returned per stock |

Scoring formula: confidence = clamp(0.35 + weeks×0.06 + tightness_bonus×0.20, 0.05, 0.95)

Source: `backend/app/analysis/patterns/three_weeks_tight.py:27-29`

### 10.2 High Tight Flag

| Constant | Value | Description |
|----------|-------|-------------|
| `_POLE_MIN_BARS` | 20 | Minimum pole duration (daily bars) |
| `_POLE_MAX_BARS` | 40 | Maximum pole duration |
| `_MIN_POLE_RETURN` | 1.0 | Minimum pole return (100% = doubling) |
| `_RECENT_POLE_BARS` | 20 | Recency window for pole weighting |
| `_MAX_POLE_CANDIDATES` | 5 | Maximum pole candidates evaluated |
| `_FLAG_MIN_BARS` | 3 | Minimum flag consolidation duration |
| `_FLAG_MAX_BARS` | 25 | Maximum flag consolidation duration |
| `_FLAG_MAX_DEPTH_PCT` | 25.0 | Maximum flag depth as % of flag high |
| `_FLAG_MAX_VOLUME_RATIO` | 1.15 | Max flag/pole volume ratio (contraction expected) |

The HTF detector requires a stock to **double** within 20–40 trading days (the "pole"), then consolidate in a shallow flag (≤25% depth) with contracting volume. This is intentionally the most extreme pattern definition in the engine.

Source: `backend/app/analysis/patterns/high_tight_flag.py:23-31`

### 10.3 Cup with Handle

| Constant | Value | Description |
|----------|-------|-------------|
| `_CUP_MIN_WEEKS` | 6 | Minimum cup duration |
| `_CUP_MAX_WEEKS` | 65 | Maximum cup duration (~15 months) |
| `_CUP_MIN_DEPTH_PCT` | 8.0 | Minimum cup depth (% from left lip) |
| `_CUP_MAX_DEPTH_PCT` | 50.0 | Maximum cup depth |
| `_CUP_MIN_RECOVERY_PCT` | 90.0 | Minimum recovery strength (right lip / left lip × 100) |
| `_MAX_CUP_CANDIDATES` | 5 | Maximum cup candidates returned |
| `_MAX_SWING_PAIR_EVALUATIONS` | 3,000 | Computational cap for deterministic runtime |
| `_HANDLE_MIN_WEEKS` | 1 | Minimum handle duration |
| `_HANDLE_MAX_WEEKS` | 5 | Maximum handle duration |
| `_HANDLE_MAX_DEPTH_PCT` | 15.0 | Maximum handle depth |
| `_HANDLE_MAX_VOLUME_RATIO` | 1.10 | Max handle/right-side volume ratio |

O'Neil's canonical cup-with-handle: 6–65 weeks duration, 8–50% depth, ≥90% recovery, followed by a 1–5 week handle with ≤15% depth and volume contraction. The `_MAX_SWING_PAIR_EVALUATIONS` cap prevents quadratic blowup in long weekly histories.

Source: `backend/app/analysis/patterns/cup_handle.py:24-34`

### 10.4 First Pullback

| Constant | Value | Description |
|----------|-------|-------------|
| `_MA_TOUCH_BAND_PCT` | 1.5 | Band width (%) around 21-day MA for touch detection |
| `_TOUCH_SEPARATION_BARS` | 5 | Minimum bars between distinct touches |
| `_MA_PERIOD_BARS` | 21 | Moving average period for reference line |
| `_ORDERLINESS_LOOKBACK_BARS` | 35 | Window for pullback orderliness scoring |
| `_PULLBACK_HIGH_LOOKBACK_BARS` | 12 | Lookback for pullback high detection |
| `_RESUMPTION_EMA_SPAN` | 10 | EMA span for resumption trigger |
| `_RESUMPTION_LOOKAHEAD_BARS` | 15 | Forward window for resumption search |
| `_RESUMPTION_MIN_VOLUME_RATIO` | 0.90 | Minimum volume ratio for resumption validation |

The first-pullback detector identifies stocks touching the 21-day MA (within 1.5% band) after a prior advance. It distinguishes first tests (higher conviction) from subsequent tests (decreasing conviction, with a -15 readiness penalty per additional test).

Source: `backend/app/analysis/patterns/first_pullback.py:21-28`

### 10.5 NR7 / Inside Day

| Constant | Value | Description |
|----------|-------|-------------|
| `_NR7_LOOKBACK_BARS` | 7 | Lookback window for narrowest-range detection |
| `_MAX_TRIGGER_CANDIDATES` | 5 | Maximum trigger candidates returned |
| `_RECENT_TRIGGER_BARS` | 20 | Recency window for trigger relevance |
| `_TRIGGER_RANGE_REFERENCE_PCT` | 4.0 | Reference range % for score normalization |
| `_VOLUME_NEUTRAL_RATIO` | 1.05 | Volume ratio ceiling for "not expanded" check |

NR7 detects the narrowest 7-day range; Inside Day detects a bar whose high/low is entirely contained within the prior bar. Combined NR7+Inside Day triggers receive the highest subtype bonus (0.18 vs 0.10/0.08). Scoring is deliberately capped (max confidence 0.78, max quality 65) reflecting the trigger-pattern nature.

Source: `backend/app/analysis/patterns/nr7_inside_day.py:23-27`

### 10.6 VCP (Legacy Wrapper)

The VCP detector wraps the legacy `VCPDetector` class and translates its output into the Setup Engine candidate model. Key scoring formula:

```
readiness_score = vcp_score × 0.7
                  + 25.0 (if ready_for_breakout)
                  - 2.0 × max(0.0, distance_to_pivot_pct)
```

The `max(0.0, ...)` clamp means stocks below the pivot (negative distance) receive zero penalty — only stocks above the pivot are penalized for being extended.

No explicit private constants beyond the minimum daily bars (120) for data sufficiency.

Source: `backend/app/analysis/patterns/vcp_wrapper.py:77-84`

### 10.7 Double Bottom

Currently a **stub detector** — returns `not_implemented` when data prerequisites are met, `insufficient_data` when not. No pattern-specific constants defined. Requires weekly bars ≥10 and daily bars ≥80 for data prerequisite checks.

Source: `backend/app/analysis/patterns/detectors/double_bottom.py`

---

## 11. Data Requirements & Feature Computation

### Data Policy

The data policy evaluates whether a stock has sufficient history for reliable pattern detection:

| Requirement | Default | Hard Floor | Status on Failure |
|-------------|---------|------------|-------------------|
| `min_daily_bars` | 252 | ≥60 | `insufficient` |
| `min_weekly_bars` | 52 | ≥20 | `insufficient` |
| `min_benchmark_bars` | 252 | ≥60 | `degraded` (if benchmark tolerance enabled) |
| `min_completed_sessions_in_current_week` | 5 | ≥1 | `degraded` (exclude current week) |

**Three status outcomes**:
- **`ok`**: All requirements met, full computation
- **`degraded`**: Missing benchmark or incomplete week; computation proceeds with reduced features
- **`insufficient`**: Critical data missing; pipeline returns null scores

When current week is incomplete, `requires_weekly_exclude_current=True` instructs the weekly resampler to drop the partial week to avoid look-ahead bias.

Source: `policy.py:24-33`, `policy.py:52-119`

### Readiness Feature Parameters

The readiness computation module uses these fixed parameters (not runtime-configurable):

| Feature | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| ATR | `atr_period` | 14 | Average True Range period |
| ATR trend | `trend_lookback` | 20 | Rolling slope window for ATR trend |
| Bollinger Bands | `bb_window` | 20 | BB calculation window |
| BB percentile | `bb_pctile_window` | 252 | Rolling percentile rank window (~1 year) |
| Volume SMA | `volume_sma_window` | 50 | Volume moving average for volume_vs_50d |
| RS lookback | `rs_lookback` | 252 | Window for RS line new-high detection |
| RS vs SPY | `rs_vs_spy_window` | 65 | Window for relative performance vs benchmark |

**Formula reference** (from `readiness.py:237-245`):
- `distance_to_pivot_pct = 100 × (close - pivot) / pivot`
- `atr14_pct = 100 × ATR14 / close`
- `atr14_pct_trend = slope(atr14_pct, lookback=20)`
- `bb_width_pct = 100 × (bb_upper - bb_lower) / bb_mid`
- `bb_width_pctile_252 = pct_rank(bb_width_pct[-1], bb_width_pct[-252:])`
- `volume_vs_50d = volume / SMA(volume, 50)`
- `rs = close / spy_close`

---

## 12. Keeping This Catalog Current

### When to Update

- **Parameter added/renamed/re-bounded** in `SETUP_ENGINE_PARAMETER_SPECS` → Update Section 4
- **Calibration profile added/modified** in `_PROFILE_BY_DETECTOR` → Update Section 9
- **New detector implemented** → Add constants to Section 10, create profile entry in Section 9
- **Gate logic changed** in `explain_builder.py` → Update Section 3
- **Scoring weights changed** in `config.py` → Update Section 2

### Source of Truth

| What | Source |
|------|--------|
| Runtime parameters | `SETUP_ENGINE_PARAMETER_SPECS` tuple in `config.py` |
| Calibration profiles | `_PROFILE_BY_DETECTOR` dict in `calibration.py` |
| Gate evaluation logic | `build_explain_payload()` in `explain_builder.py` |
| Scoring weights | `SETUP_SCORE_WEIGHTS`, `CANDIDATE_SETUP_SCORE_WEIGHTS` in `config.py` |
| Rating rules | `calculate_rating()` in `setup_engine_screener.py` |
| Operational flags | `compute_operational_flags()` in `operational_flags.py` |

### After Parameter Changes

1. Run `make gates` to verify all quality gates pass
2. Regenerate golden snapshots if needed: `make golden-update`
3. Update this catalog to reflect the new values

### Future Enhancement

Auto-generate Sections 1, 4, and 9 tables from `SETUP_ENGINE_PARAMETER_SPECS` and `_PROFILE_BY_DETECTOR` at build time, reducing the risk of documentation drift.
