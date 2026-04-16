# SE-H2: Staged Rollout Plan with Feature Flags and Observability Metrics

> **Operator playbook** for safely enabling the Setup Engine screener, monitoring its behavior at each stage, and rolling back if needed. Designed for a self-hosted deployment where "rollout" means progressively building trust — not percentage-based traffic splitting.

## Table of Contents

1. [Rollout Overview](#1-rollout-overview)
2. [Infrastructure Inventory](#2-infrastructure-inventory)
3. [Phase 0 — Pre-Flight Validation](#3-phase-0--pre-flight-validation)
4. [Phase 1 — Shadow Observation](#4-phase-1--shadow-observation-1-2-weeks)
5. [Phase 2 — Conservative Enablement](#5-phase-2--conservative-enablement-2-4-weeks)
6. [Phase 3 — Parameter Tuning](#6-phase-3--parameter-tuning-ongoing)
7. [Phase 4 — Full Confidence](#7-phase-4--full-confidence-steady-state)
8. [Emergency & Rollback Procedures](#8-emergency--rollback-procedures)
9. [Metric Reference Card](#9-metric-reference-card)
10. [Keeping This Plan Current](#10-keeping-this-plan-current)

### Related Design Docs

- [SE-H1: Parameter Catalog](se_h1_parameter_catalog.md) — Threshold defaults, bounds, guardrails, recalibration guide
- [SE-A3: Parameter Governance](se_a3_parameter_governance.md) — Parameter dataclass and validation architecture
- [SE-A4: Data Requirements Policy](se_a4_data_requirements_policy.md) — Data sufficiency semantics

### Canonical Source Files

| File | Role |
|------|------|
| `backend/app/config/settings.py` | Feature flag definition (singleton at `:221`) |
| `backend/app/scanners/scan_orchestrator.py` | Kill-switch filtering logic |
| `backend/app/tasks/scan_tasks.py` | Post-scan SE telemetry function |
| `backend/app/scanners/setup_engine_screener.py` | Per-symbol timing + rating logic |
| `backend/app/celery_app.py` | Startup warning when disabled |
| `backend/app/analysis/patterns/config.py` | 18 parameters, specs, guardrails |
| `Makefile` | Quality gate targets |
| `docs/release-checklist.md` | Pre-merge verification |

---

## 1. Rollout Overview

### Key Context

- **Self-hosted, not SaaS.** There is no canary deployment or percentage-based rollout. "Rollout" means the operator progressively trusts SE-inclusive results over a series of phases.
- **Composite weight redistribution.** Enabling SE changes composite scoring from 5-screener equal-weight (1/5 = 20% each) to 6-screener equal-weight (1/6 ≈ 16.7% each). Every existing screener's influence decreases by ~3.3 percentage points.
- **No technical shadow mode.** When `SETUP_ENGINE_ENABLED=true`, SE contributes to composite scores immediately. "Shadow observation" (Phase 1) is a behavioral discipline — the operator does not act on SE-influenced rankings — not a system isolation feature.
- **Zero additional API calls.** SE sets `needs_benchmark=True` in its `DataRequirements` (`setup_engine_screener.py:69`), but so do Minervini and CANSLIM. The orchestrator merges all requirements via `DataRequirements.merge_all` (`base_screener.py:55`) and fetches once. Enabling SE adds compute, not network calls.

### Phase Summary

| Phase | Duration | Flag State | Objective |
|-------|----------|------------|-----------|
| **0 — Pre-Flight** | 1 day | `false` | Validate infrastructure: gates green, Docker builds, flag gates correctly |
| **1 — Shadow Observation** | 1–2 weeks | `true` | Monitor SE telemetry without acting on SE-influenced results |
| **2 — Conservative Enablement** | 2–4 weeks | `true` | Begin using SE-inclusive results; capture baseline metrics |
| **3 — Parameter Tuning** | Ongoing | `true` | Optimize thresholds using SE-H1 recalibration guide |
| **4 — Full Confidence** | Steady state | `true` | SE fully trusted; maintenance monitoring only |

---

## 2. Infrastructure Inventory

Everything needed for rollout already exists. No new code, configuration, or infrastructure is required.

| Concern | Mechanism | Location |
|---------|-----------|----------|
| Kill switch | `SETUP_ENGINE_ENABLED` env var → `settings.setup_engine_enabled` | `settings.py:63` |
| Orchestrator filtering | Silent removal of `setup_engine` from screener list | `scan_orchestrator.py:88-90` |
| Startup notification | Celery logs warning when disabled | `celery_app.py:71-72` |
| Score distribution telemetry | Post-scan log line: n, min, max, mean, median, ready | `scan_tasks.py:183-219` |
| Per-symbol timing | INFO-level breakdown: prep/detectors/readiness/total ms, errors, no_data | `setup_engine_screener.py:277-282` |
| Parameter overrides | `build_setup_engine_parameters(overrides={...})` | `config.py:34-71` |
| Guardrail validation | 4 cross-parameter validators | `config.py` |
| Quality gates | `make gates` (5 gate suites, Gate 4 advisory) | `Makefile` |
| Golden regression | Snapshot-pinned detector, aggregator, scanner outputs | `Makefile:64-67` |
| Release checklist | Manual + automated pre-merge checks | `docs/release-checklist.md` |
| Health probes | `/livez` (liveness), `/readyz` (readiness) | `main.py:172, 178` |
| Frontend SE drawer | Rich gate/flag visualization per stock | `SetupEngineDrawer.jsx` |

---

## 3. Phase 0 — Pre-Flight Validation

**Duration:** 1 day
**Flag state:** `SETUP_ENGINE_ENABLED=false`

### Checklist

1. **Run all quality gates:**
   ```bash
   make all
   ```
   All 5 gates must pass (Gate 4 is advisory but review any regressions). `gate-check` verifies no SE test files are unassigned.

2. **Walk through the release checklist:**
   ```bash
   cat docs/release-checklist.md
   ```
   Complete every item. The checklist covers automated gates, manual verification, and deployment readiness.

3. **Verify feature flag gating:**
   - Confirm `.env` has `SETUP_ENGINE_ENABLED=false`
   - Start backend + Celery workers
   - Check Celery startup logs for: `SetupEngine scanner disabled via SETUP_ENGINE_ENABLED=false`
   - Run a scan that includes `setup_engine` in screener selection
   - Verify SE is silently filtered — no SE scores in results, no errors

4. **Docker build verification** (if using Docker deployment):
   ```bash
   docker-compose build
   docker-compose up -d
   # Verify /readyz returns healthy
   curl http://localhost:8000/api/readyz
   ```

### Success Criteria

| Check | Expected |
|-------|----------|
| `make all` | All gates green |
| Celery startup log | Warning message present |
| Scan with SE requested + flag=false | SE silently excluded, no errors |
| Docker build + `/readyz` | Healthy |

### Exit → Phase 1

All pre-flight checks pass. No code changes needed.

---

## 4. Phase 1 — Shadow Observation (1–2 weeks)

**Duration:** 1–2 weeks
**Flag state:** `SETUP_ENGINE_ENABLED=true`

### Enabling SE

1. Set `SETUP_ENGINE_ENABLED=true` in `.env` (or `.env.docker`)
2. **Restart ALL processes** — this is critical:
   - **Local:** Restart uvicorn + both Celery workers + Celery Beat
   - **Docker:** `docker-compose restart backend celery-worker celery-datafetch celery-beat`
   - **Why:** `settings = Settings()` is a module-level singleton (`settings.py:221`) loaded once at import time. Each process holds its own frozen copy. Changing `.env` without restarting has no effect.
3. Verify Celery startup logs do **not** show the disabled warning

### Important Caveat

SE IS contributing to composite scores. Rankings WILL shift compared to pre-SE scans. "Shadow observation" means:
- Run scans as normal
- Review SE telemetry in Celery logs
- **Do not act on SE-influenced rankings for trading decisions**
- Compare results with and without SE selected (run two scans with different screener selections) to understand the impact

### Monitoring Metrics

| Metric | Source | Healthy Range | Warning Threshold |
|--------|--------|--------------|-------------------|
| SE coverage rate | Telemetry `n` vs universe size | >90% of universe scored | <80% coverage |
| Timing p95 | Per-symbol log `total_ms` | <2000ms | >2000ms |
| Error rate | Per-symbol log `errors` count | <1% of symbols | >5% of symbols |
| Score distribution | Telemetry min/max/mean/median | mean 30–60, reasonable spread | Degenerate: all 0 or all >90 |
| Ready rate | Telemetry `ready` / `n` | 5–15% | 0% (nothing qualifies) or >50% (too permissive) |
| Top-50 composite impact | Compare SE-on vs SE-off scans | Shift <5 pts for top 50 | Top-50 dramatically reorders |

### How to Read the Telemetry

After each scan completes, the post-scan pipeline logs a line like:

```
SE score distribution [scan-abc123]: n=487 min=0.0 max=89.3 mean=38.2 median=35.7 ready=42
```

Source: `scan_tasks.py:213-216`

Per-symbol timing appears for every stock:

```
SE timing AAPL: prep=12.3ms detectors=45.6ms readiness=8.1ms total=66.0ms score=72.5 errors=0 no_data=0
```

Source: `setup_engine_screener.py:277-282`

### Advancement Criteria

- [ ] 1–2 weeks of scans with no warning-level metrics
- [ ] Error rate consistently <1%
- [ ] Score distribution is non-degenerate (mean between 30–60, some spread)
- [ ] Ready rate is plausible (5–15%)
- [ ] No unresolved Celery worker crashes or OOM events

---

## 5. Phase 2 — Conservative Enablement (2–4 weeks)

**Duration:** 2–4 weeks
**Flag state:** `SETUP_ENGINE_ENABLED=true`

### Behavioral Change

Begin using SE-inclusive results for screening decisions. This phase adds deeper quality metrics beyond the Phase 1 health checks.

### Additional Monitoring Metrics

| Metric | Source | Healthy Range | Warning Threshold |
|--------|--------|--------------|-------------------|
| Rating distribution | SQL query (below) | Pyramid: StrongBuy < Buy < Watch < Pass | Inverted pyramid or single-rating dominance |
| Gate failure distribution | Aggregate `failed_checks` | No single gate >60% of failures | One gate dominates (>60%) |
| Operational flag frequency | Aggregate `invalidation_flags` | 10–30% of stocks flagged | 0% (flags not firing) or >60% (too aggressive) |
| Daily ranking stability | Consecutive scan comparison | >80% top-50 overlap day-to-day | <60% overlap (excessive churn) |
| Pattern diversity | Count distinct `pattern_primary` | 3+ pattern types detected | Only 1 pattern type |

### Baseline Snapshot Procedure

Capture this baseline **before** advancing to Phase 3. This becomes the reference point for parameter tuning.

**Step 1 — Telemetry snapshot.** After a full-universe scan, grep Celery logs for the SE distribution line:

```bash
# Local: check celery worker logs
grep "SE score distribution" /path/to/celery-worker.log | tail -1
```

Record the `n`, `min`, `max`, `mean`, `median`, `ready` values.

**Step 2 — Rating distribution query.** Run against the database:

```sql
SELECT json_extract(details, '$.details.screeners.setup_engine.rating') AS se_rating,
       COUNT(*) AS cnt
FROM scan_results
WHERE scan_id = '<latest-scan-id>'
  AND json_extract(details, '$.details.screeners.setup_engine.rating') IS NOT NULL
GROUP BY se_rating
ORDER BY cnt DESC;
```

Expected pyramid shape:
```
Pass        ~350
Watch       ~100
Buy         ~30
Strong Buy  ~10
```

**Step 3 — Manual chart verification.** In the frontend:
1. Sort results by SE `setup_score` descending
2. Open the top 10 stocks' SetupEngineDrawer
3. Verify the detected `pattern_primary` (e.g., "cup_with_handle", "flat_base") matches the visible chart pattern
4. Note any obvious misclassifications

**Step 4 — Save the baseline.** Store the log snippet, SQL output, and manual verification notes for comparison after parameter tuning in Phase 3.

### Advancement Criteria

- [ ] Baseline snapshot captured and saved
- [ ] 2–4 weeks of stable operation with no warning-level metrics
- [ ] Rating distribution follows expected pyramid shape
- [ ] Pattern diversity ≥ 3 types
- [ ] Top-50 overlap between consecutive daily scans >80%

---

## 6. Phase 3 — Parameter Tuning (Ongoing)

**Duration:** Ongoing (typically 1–2 weeks of active tuning, then periodic revisits)
**Flag state:** `SETUP_ENGINE_ENABLED=true`

### Reference

Parameter tuning follows the **Recalibration Guide** in SE-H1 Section 6. All 18 runtime-tunable parameters are documented in the **Configurable Threshold Catalog** (SE-H1 Section 4).

### Tuning Protocol

1. **Change one parameter at a time.** Multi-parameter changes make it impossible to attribute effects.

2. **Run quality gates after each change:**
   ```bash
   make gates
   ```
   All gates must remain green.

3. **If golden snapshots change:** The expected-output snapshots are pinned to specific parameter values. After a threshold change:
   ```bash
   make golden-update
   ```
   Review the diff — changes should be explainable by the parameter you adjusted.

4. **Run a full-universe scan** and compare telemetry against your Phase 2 baseline.

### Common Tuning Scenarios

**Increasing throughput** (more stocks qualify as "ready"):
- Lower `readiness_score_ready_min_pct` (default: 70.0)
- Lower `quality_score_min_pct` (default: 60.0)
- Widen the early zone: increase `early_zone_distance_to_pivot_pct_max` (default: 3.0)
- Watch for: ready rate increasing beyond 20%, rating pyramid inverting

**Increasing selectivity** (fewer, higher-conviction signals):
- Raise `readiness_score_ready_min_pct` above 70.0
- Raise `setup_score_min_pct` above 65.0
- Tighten ATR cap: lower `atr14_pct_max_for_ready` (default: 8.0)
- Watch for: ready rate dropping to 0%, only 1–2 stocks qualifying

**Market regime adaptation:**
- Bull market: Consider raising `context_rs_rating_min` (default: 50.0) to filter laggards more aggressively
- Volatile market: Raise `atr14_pct_max_for_ready` to avoid filtering out volatile leaders
- Narrow market: Lower `quality_score_min_pct` to catch narrower bases

> **Warning**: Parameters interact — see SE-H1 Section 5 "Parameter Interactions" before tuning multiple knobs simultaneously.

### Advancement Criteria

- [ ] Operator is satisfied with the current parameter set
- [ ] Golden snapshots are stable after `make golden-update`
- [ ] Telemetry metrics remain within healthy ranges from the Metric Reference Card (Section 9)

---

## 7. Phase 4 — Full Confidence (Steady State)

**Duration:** Ongoing
**Flag state:** `SETUP_ENGINE_ENABLED=true`

### Operational Posture

SE is fully trusted as a primary screening signal alongside Minervini, CANSLIM, and other screeners.

### Maintenance Monitoring

| Activity | Frequency | Action |
|----------|-----------|--------|
| `make gates` | Before every deployment | All 5 gates must pass |
| Review SE telemetry | After each scan | Spot-check distribution log line |
| Distribution drift check | Monthly | Compare current telemetry against Phase 2 baseline |
| Golden snapshot review | After code changes | `make golden-update` + review diff |

### Regression Detection

If golden snapshots change **without intentional code or parameter changes**, investigate:

1. Was a dependency updated (pandas, numpy)?
2. Did data policy or data source change?
3. Is there a floating-point precision issue from a platform change?

Any unexplained golden drift warrants returning to Phase 2 observation until the cause is identified.

---

## 8. Emergency & Rollback Procedures

### Immediate Kill-Switch (Any Phase)

**Step 1:** Set the flag:
```bash
# In .env (local) or .env.docker (Docker)
SETUP_ENGINE_ENABLED=false
```

**Step 2:** Restart ALL processes — this is **critical** because `settings = Settings()` is a module-level singleton (`settings.py:221`) loaded once at import time. Each process holds its own frozen copy:

```bash
# Local development
# Restart uvicorn (Ctrl-C and re-run)
# Restart both Celery workers + Beat (Ctrl-C start_celery.sh and re-run)

# Docker
docker-compose restart backend celery-worker celery-datafetch celery-beat
```

**Step 3:** Verify the kill-switch took effect:
```bash
# Check Celery startup logs for:
# "SetupEngine scanner disabled via SETUP_ENGINE_ENABLED=false"
```

**Step 4:** New scans will exclude SE. Composite scoring reverts to N–1 screener equal-weight average.

### What "Disabled" Means

- `scan_orchestrator.py:88-90`: Silently removes `setup_engine` from the screener list. No per-symbol warnings, no errors.
- The frontend "Setup" chip still appears in the screener selector, but the backend ignores it.
- No database migration or cleanup required. SE data in existing scan results is inert JSON — it doesn't affect anything when SE is disabled.

### Historical Data Caveat

Scans run while SE was enabled retain their 6-screener composite scores. New scans after disabling will have 5-screener composites. **Comparing raw composite scores across this boundary is misleading** — compare ranks (top-N overlap), not raw numbers.

### Diagnostic Decision Tree

After disabling SE, determine the root cause before deciding next steps:

```
Is the issue SE-specific or pipeline-wide?
├── Pipeline-wide → SE is not the cause; investigate other screeners
└── SE-specific
    ├── Is it timing? (scans too slow)
    │   ├── Check Gate 4 performance budgets
    │   └── Review per-symbol timing p95 (setup_engine_screener.py logs)
    ├── Is it score quality? (bad rankings)
    │   ├── Try parameter tuning first (Phase 3 actions)
    │   └── If tuning fails → keep disabled, file bug
    └── Is it errors/crashes?
        └── Keep disabled, file bug with stack trace from per-symbol error log
```

### Parameter-Only Rollback (Non-Emergency)

If the issue is score quality rather than errors, try reverting parameters before disabling SE entirely:

1. Remove parameter overrides to revert to defaults:
   - Delete any `setup_engine_parameters` overrides from scan criteria
   - Default values are documented in SE-H1 Section 4

2. Or restore known-good override values from your Phase 2 baseline.

3. Run `make golden-update` to regenerate snapshots matching reverted parameters.

4. Run a full-universe scan and compare against baseline.

### Data Recovery (Extremely Unlikely)

The SE architecture stores all data as isolated JSON within existing `scan_results` rows. There is no separate SE table. If you need to clean up:

1. Disable SE via feature flag
2. Clean up any orphaned scans:
   ```bash
   cd backend && source venv/bin/activate
   python scripts/cleanup_orphaned_scans.py
   ```
3. Re-scan with SE disabled

---

## 9. Metric Reference Card

Consolidated single-page reference for all rollout monitoring metrics.

| Metric | Phase | Source | Normal Range | Action if Out of Range |
|--------|-------|--------|-------------|------------------------|
| SE coverage rate | 1+ | Telemetry `n` / universe | >90% | Check data policy thresholds; review `insufficient_data` count in per-symbol logs |
| Timing p95 | 1+ | Per-symbol log `total_ms` | <2000ms | Check which phase is slow (prep/detectors/readiness); review Gate 4 budgets |
| Error rate | 1+ | Per-symbol log `errors` | <1% | Review stack traces; check for data format issues |
| Score mean | 1+ | Telemetry `mean` | 30–60 | Mean <20: calibration too strict; mean >70: calibration too permissive. Review SE-H1 Section 6 |
| Ready rate | 1+ | Telemetry `ready` / `n` | 5–15% | 0%: readiness gates too strict, lower `readiness_score_ready_min_pct`; >30%: too permissive, raise it |
| Rating pyramid | 2+ | SQL query (Section 5) | StrongBuy < Buy < Watch < Pass | Inverted: lower score thresholds in `calculate_rating` logic |
| Gate failure distribution | 2+ | Aggregate `failed_checks` from SE drawer | No single gate >60% | If one gate dominates failures, review that gate's threshold (see SE-H1 Section 3) |
| Top-50 overlap | 2+ | Compare consecutive scans | >80% top-50 overlap day-to-day | <60%: check data freshness, cache TTL, or volatile parameter settings |
| Daily stability | 2+ | Consecutive scan comparison | >80% | Excessive churn: review `atr14_pct_max_for_ready` and volume filters |
| Pattern diversity | 2+ | Count distinct `pattern_primary` | 3+ types detected | Only 1 type: review detector implementations; check data sufficiency for other pattern types |

### Telemetry Log Formats

**Post-scan distribution** (one per scan):
```
SE score distribution [<scan_id>]: n=<count> min=<min> max=<max> mean=<mean> median=<median> ready=<ready_count>
```
Source: `scan_tasks.py:213-216`

**Per-symbol timing** (one per stock):
```
SE timing <SYMBOL>: prep=<ms>ms detectors=<ms>ms readiness=<ms>ms total=<ms>ms score=<score> errors=<n> no_data=<n>
```
Source: `setup_engine_screener.py:277-282`

**Celery startup** (one per worker):
```
SetupEngine scanner disabled via SETUP_ENGINE_ENABLED=false
```
Source: `celery_app.py:71-72` (only when disabled)

---

## 10. Keeping This Plan Current

Update this document when:

- **New parameters added** — add to the Metric Reference Card if they affect observability
- **New quality gates added** — update the Infrastructure Inventory table and Phase 0 checklist
- **Composite scoring algorithm changes** — revise the "composite weight redistribution" note in Section 1 and the historical data caveat in Section 8
- **New deployment modes added** — update restart procedures in Section 8
- **Phase durations prove wrong** — adjust recommended durations based on operational experience
- **New telemetry added** — update log format examples in Section 9

### Document Provenance

| Field | Value |
|-------|-------|
| Author | SE-H2 task |
| Created | 2026-02-22 |
| SE-H1 version referenced | 2026-02-19 (1fe0236d) |
| Codebase commit | Verified against current `main` |
