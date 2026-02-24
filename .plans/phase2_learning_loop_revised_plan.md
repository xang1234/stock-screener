# Phase 2 Revised Plan: Learning Loop (Auto‑Evaluation of Setups)

> **Goal:** Turn StockScreener from “scan → eyeball charts” into a **self-measuring Setup Engine** that:
> - automatically tracks how each setup behaved **after it appeared**,
> - lets you answer **which patterns + filters historically had the best follow‑through**,
> - and (optionally) uses that data to **calibrate scoring weights** in a controlled, reproducible way.
>
> This is **descriptive analytics** (not trading advice). The focus is correctness, reproducibility, and usability for a swing/position trading workflow.

---

## 0) What changed vs the original Phase 2 plan

The initial idea (“store forward returns on each scan result and tune weights”) is directionally right, but the **current repo architecture** strongly suggests a better foundation:

- You now have a **daily Feature Store** (`feature_runs`, `stock_feature_daily`, pointer publishing) and a dedicated **BuildDailyFeatureSnapshotUseCase** invoked from Celery (`app.interfaces.tasks.feature_store_tasks.build_daily_snapshot`).
- Scans are increasingly **bound to a feature_run_id**, and the Feature Store is already the natural “dataset” for research and analytics.

### Revised stance
1. **Compute outcomes primarily on the Feature Store dataset**, not per ad‑hoc scan row.
2. Treat “Learning Loop” as its own bounded context: an **Outcome Engine + Evaluation Lab**.
3. Keep everything **idempotent** and **versioned**, so comparisons across code changes remain meaningful.

---

## 1) Architecture overview

### 1.1 Core pipeline
**(A) Feature snapshot is produced** (already exists)
- Daily Celery beat runs `build_daily_snapshot` → creates a `FeatureRun` + `StockFeatureDaily` rows.

**(B) Outcome Engine produces labels for that snapshot**
- A scheduled job computes forward returns, excursions (MFE/MAE), and follow‑through proxies for each `(run_id, symbol)`.

**(C) Evaluation Lab aggregates**
- Provides APIs/UI to answer:
  - “Which pattern families outperformed over 20 trading days?”
  - “Which preset filters had the best hit‑rate / median return / MFE?”
  - “Which setups tend to fail fast (high MAE) even if average return is OK?”

**(D) Optional Calibration**
- Uses labeled dataset to fit weight sets (in a controlled, reversible way).

### 1.2 Design principles (robustness)
- **No look‑ahead:** entry and horizons defined using trading days, with a consistent time convention.
- **Corporate actions aware:** use *adjusted* prices (or store both raw/adjusted).
- **Idempotent & resumable:** outcomes can be recomputed safely; partial horizons fill as data becomes available.
- **Reproducible:** store `code_version`, `outcome_version`, and configuration hashes.
- **Separation of concerns:** scanning produces features; outcome engine produces labels; evaluation reads both.

---

## 2) Data model additions

### 2.1 New table: `feature_outcomes_daily` (wide, one row per (run_id, symbol))
Create a new table rather than stuffing outcomes into `details_json`. Reasons:
- forward-performance metrics are numeric and queried often → should be indexed
- avoids bloating `details_json`
- supports partial filling (5d ready before 60d ready)

**Schema (recommended):**
- `run_id` (FK → feature_runs.id, PK part)
- `symbol` (PK part)
- `as_of_date` (denormalized; query convenience)
- **Entry prices (store both)**
  - `entry_close` (raw)
  - `entry_adj_close` (preferred baseline)
  - `entry_next_open` (raw; optional)
  - `entry_next_adj_open` (optional)
  - `entry_basis` TEXT (e.g., `"close"` default, but allow `"next_open"` for alternate evaluation)
  - `adj_factor` REAL (entry_adj_close / entry_close) so you can adjust OHLC consistently
- **Forward returns (adj-based by default)**
  - `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d`
- **Excursions**
  - `mfe_5d`, `mfe_10d`, `mfe_20d`, `mfe_60d`
  - `mae_5d`, `mae_10d`, `mae_20d`, `mae_60d`
- **Follow-through / behavior flags** (booleans as INTEGER 0/1)
  - `made_new_20d_high_within_10d`
  - `made_new_50d_high_within_20d`
  - `closed_above_entry_5d` (simple “did it work quickly?” proxy)
  - `hit_stop_proxy_10d` (see definitions below)
- **Dates for interpretability**
  - `d5_date`, `d10_date`, `d20_date`, `d60_date`
  - `first_breakout_date` (optional, if you define breakout using pivot from Setup Engine)
- **Metadata**
  - `computed_at`
  - `outcome_version` (TEXT, e.g. `"v1"`; bump when definitions change)
  - `notes` (TEXT or JSON; for missing-data reasons, delist, halt, etc.)

**Indexes**
- `(run_id)` (already implied by PK but keep explicit index in SQLite migrations)
- `(as_of_date)` for time series research
- `(symbol, as_of_date)` to fetch a ticker’s timeline quickly
- Optionally `(ret_20d)` if you want “top follow-through” queries

### 2.2 Optional: `evaluation_cache` (materialized aggregates)
SQLite + JSON filters can get slow when you start doing repeated aggregate queries across months of data.

Add a caching table to store precomputed aggregate metrics, keyed by:
- `cache_key` (hash of filter spec + horizon + date range + benchmark selection)
- `created_at`
- `payload_json` (aggregates, counts, histograms)

This keeps the UI snappy and makes “preset leaderboards” instant.

### 2.3 Optional: `calibrated_weight_sets`
If you add auto-calibration, do it safely:
- `id`, `name`, `created_at`
- `target_metric` (e.g. `"ret_20d"`, `"mfe_20d_minus_mae_20d"`, `"hit_rate_20d"`)
- `train_range`, `test_range` (date)
- `feature_version` / `outcome_version`
- `weights_json` (mapping feature → weight)
- `metrics_json` (in-sample + out-of-sample performance summary)
- `is_active` flag (only 1 active per “mode”, if you want)

---

## 3) Precise outcome definitions (so the data is trustworthy)

### 3.1 Trading-day horizons
All horizons are **trading days**, not calendar days.

Implementation approach (consistent with existing codebase):
- Use `pandas_market_calendars` NYSE calendar (already used in `BuildDailyFeatureSnapshotUseCase`) to compute next trading sessions for US stocks.
- Store the actual dates (`d5_date`, etc.) so the results remain interpretable and reproducible even if calendars change.

### 3.2 Entry price convention
Provide two evaluation modes; store both entry prices so you can pick later:

1) **Close Entry (default for EOD scanning)**
- `entry_adj_close = stock_prices.adj_close` on `as_of_date`
- if `adj_close` missing, fallback to `close` and set `adj_factor = 1.0` with a warning note

2) **Next Open Entry (more conservative / tradeable if scan runs after close)**
- `entry_next_adj_open = next_day.open * next_day_adj_factor`
- next_day_adj_factor can be approximated as `(next_day.adj_close / next_day.close)` when available, or `(entry_adj_close / entry_close)` if not.

You can calculate all outcomes from a chosen `entry_basis`, but storing both lets you compare how sensitive the “edge” is to entry assumptions.

### 3.3 Returns (adj-based)
For horizon H:
- `ret_H = (adj_close_on_H - entry_adj_price) / entry_adj_price`

Where:
- `adj_close_on_H` is the adjusted close on the H-th trading day after `as_of_date`.

### 3.4 MFE/MAE (adj-consistent)
For horizon H:
- Build adjusted high/low:
  - `high_adj = high * adj_factor_for_day`
  - `low_adj = low * adj_factor_for_day`
- `mfe_H = max(high_adj over days 1..H) / entry_adj_price - 1`
- `mae_H = min(low_adj over days 1..H) / entry_adj_price - 1`

### 3.5 Follow-through proxies (actionable evaluation beyond raw returns)
These are optional but **highly useful** for setup-based trading research.

**(A) New-high follow-through**
- `made_new_20d_high_within_10d = 1` if within the first 10 trading days after as_of_date,
  the stock’s adjusted **high** exceeds the prior 20-trading-day adjusted high (computed using data up to as_of_date).
- Similar for `made_new_50d_high_within_20d`.

This proxies: “did it actually act like a leader and push to new highs soon?”

**(B) Stop proxy**
A simple descriptive “pain” metric:
- Define `stop_proxy_pct` (config, e.g., 7–10% below entry; you can keep it as analytics only)
- `hit_stop_proxy_10d = 1` if MAE_10d <= -stop_proxy_pct

**(C) Time-to-breakout (requires Setup Engine pivot)**
If Phase 1 stores `pivot_price` in `details_json`:
- `first_breakout_date` = first trading day where `high_adj >= pivot_adj` (or `close_adj >= pivot_adj`, your choice)
- Add `days_to_breakout` integer

This is **extremely compelling** for evaluating “pre-breakout readiness” features.

---

## 4) Outcome Engine implementation plan

### 4.1 New bounded context modules (backend)
Add a dedicated module set so this doesn’t pollute scanning logic:

```
backend/app/domain/evaluation/
  models.py          # OutcomeRow, OutcomeConfig, AggregateMetrics
  ports.py           # OutcomeRepository, PriceRepository, EvalCacheRepository
  calculations.py    # pure functions (returns/mfe/mae/follow-through)
  dq.py              # data-quality checks for outcomes
backend/app/use_cases/evaluation/
  compute_outcomes_for_run.py
  backfill_outcomes.py
  aggregate_performance.py
  leaderboard_presets.py
  calibrate_weights.py (optional)
backend/app/infra/db/models/
  feature_outcomes.py
backend/app/infra/db/repositories/
  outcomes_repo.py
  price_repo.py (adapter over stock_prices queries)
backend/app/interfaces/tasks/
  evaluation_tasks.py
backend/app/db_migrations/
  outcomes_migration.py
```

### 4.2 Celery scheduling (beat)
Add a scheduled task **after** the daily data refresh has updated `stock_prices`:

- `daily-compute-outcomes`:
  - runs nightly (or morning) Mon–Fri
  - does **not** need the serialized data_fetch lock (no external API calls)
  - routes to the general `celery` queue (parallelizable)

**Job algorithm**
1. Find FeatureRuns eligible for outcome computation:
   - status is `published` (or at least `completed`, configurable)
   - outcomes missing OR outdated outcome_version
2. For each run:
   - determine which horizons have enough future data available (e.g., run_date + 5 trading days ≤ latest_price_date)
3. Fetch all needed `stock_prices` for symbols in that run in **one range query**:
   - date range = `[as_of_date .. as_of_date + max_horizon_window]`
   - symbols = `SELECT symbol FROM feature_run_universe_symbols WHERE run_id = ?`
4. Compute outcomes in vectorized Python (pandas groupby) or efficient loops.
5. Upsert rows into `feature_outcomes_daily` (idempotent).
6. Run outcome DQ checks; if catastrophic missingness, write warnings and skip caching.

### 4.3 Idempotency & versioning
- Every computation writes `outcome_version="v1"` and `computed_at`.
- If `outcome_version` changes in code, the task recomputes and overwrites outcomes (or writes to a new versioned table if you prefer immutability).
- Use `INSERT ... ON CONFLICT(run_id, symbol) DO UPDATE` in SQLite for upserts.

### 4.4 Performance notes
- Avoid per-symbol DB queries. Always fetch in bulk for a run.
- Chunk extremely large runs (e.g., 9,000 symbols) by symbol list to reduce memory spikes:
  - compute outcomes for 500–1000 symbols at a time
  - upsert chunk results then continue
- Add pragmatic limits:
  - only compute 60d outcomes for runs older than 60 trading days
  - compute 5/10/20 daily for recent runs

---

## 5) Evaluation Lab features (what makes this compelling)

### 5.1 “Filter → performance” explorer (API + UI)
Add an endpoint that accepts:
- date range (e.g., last 252 trading days)
- horizon (5/10/20/60)
- feature filters (reuse your existing filter spec machinery)
- optional benchmark (SPY) and regime split

Returns:
- sample size, coverage %
- mean/median return
- hit rate (% positive)
- p25/p50/p75 return
- mean MFE / mean MAE
- “fast failure” rate (hit_stop_proxy_10d)
- follow-through rates (new highs)

This is the single most useful “research lever” for a swing trader.

### 5.2 Preset leaderboards (automatic)
Use your existing `filter_presets` table:
- nightly compute metrics for each preset:
  - last 3 months / 6 months / 12 months
  - ret_20d median, hit_rate_20d, mae_10d, etc.
- store results in `evaluation_cache` to avoid recomputing constantly
- expose in UI as “Preset Performance” with sorting and drill-down

### 5.3 Pattern family performance (from Setup Engine)
Once Phase 1 stores pattern flags and readiness metrics:
- compute performance by:
  - `pattern_family` (VCP / 3WT / HTF / CupHandle / etc.)
  - readiness buckets (e.g., distance_to_pivot deciles)
  - leadership buckets (RS rating, RS line new highs)
- show “What worked recently?” and “What works in general?”

This closes the loop: the scanner learns which “setups” in *your own definitions* historically behaved best.

### 5.4 Market regime slicing (makes results credible)
Define a lightweight `market_regime` per day (for example):
- SPY above/below 200MA
- SPY 50MA slope up/down
- breadth regime using existing breadth indicators

Then every evaluation can be split:
- bull regime vs bear regime
- high volatility vs low volatility

This avoids misleading averages and is very compelling in practice.

---

## 6) Score calibration (optional, but do it safely)

### 6.1 What not to do
- Don’t silently “auto-change” weights used by production scans.
- Don’t fit complex models without controlling leakage and sample size.

### 6.2 Recommended approach
Create “weight sets” as artifacts:
- baseline weights (hand-authored)
- calibrated weights (derived)
- user can select active weight set (or just compare)

**Calibration method options**
- **Regularized regression** on a target metric (e.g., ret_20d)
- **Logistic regression** for a binary target (e.g., `made_new_20d_high_within_10d`)
- Keep features limited to:
  - screener scores (minervini, canslim, setup_engine score)
  - a few readiness metrics (distance_to_pivot, atr_pct, bb_squeeze, volume_dryup)
  - leadership (rs_rating, group_rank)

**Guardrails**
- minimum sample size per regime
- train/test split by date (walk-forward)
- coefficient clipping + shrinkage
- report in-sample vs out-of-sample metrics
- store everything in `calibrated_weight_sets` with versions

---

## 7) Data quality & reliability upgrades (critical for trust)

### 7.1 Outcome DQ checks (add a DQ module like feature_store.quality)
Examples:
- missing price coverage ratio per run (critical if > X%)
- entry price null rate
- extreme return outliers (warn if abs(ret_20d) > 200% and not split-adjusted)
- horizon date mismatch (calendar logic correctness)
- symbol mapping anomalies (ticker changed; missing future data)

Outcomes that fail DQ should be flagged with `notes` and optionally excluded from aggregates.

### 7.2 Reproducibility
- Store `outcome_version` and `feature code_version` (already on feature_runs).
- Store config used for outcomes (entry_basis default, stop_proxy_pct, new-high windows) either:
  - in `feature_runs.config_json`, or
  - in `feature_outcomes_daily.notes/config_json`

---

## 8) Implementation milestones (suggested)

### Milestone 1 — Outcome storage + compute
- Add `feature_outcomes_daily` model + idempotent migration.
- Implement `ComputeOutcomesForRunUseCase` (pure-ish + repositories).
- Add Celery task + beat schedule.
- Verify correctness on a small subset of symbols.

### Milestone 2 — Evaluation Explorer API
- Join `stock_feature_daily` + `feature_outcomes_daily`.
- Add aggregates endpoint with date range + horizon + filters.
- Add caching for common queries.

### Milestone 3 — Preset Leaderboards
- Iterate all filter presets nightly.
- Store results in `evaluation_cache`.
- Add UI page with drilldown.

### Milestone 4 — Pattern performance (ties to Phase 1)
- Aggregate by pattern family flags and readiness buckets.

### Milestone 5 — Calibration (optional)
- Add weight set artifacts.
- Add offline calibration job + reporting.
- Never auto-switch weights without explicit user action.

---

## 9) Notes on “scan results” integration (don’t duplicate work)
Where possible:
- Use `scan.feature_run_id` to show outcomes for scan rows by joining to `feature_outcomes_daily`.
- For legacy scans without `feature_run_id`, either:
  - compute outcomes on demand (slow), or
  - backfill a feature run for that scan date + universe and bind it.

This keeps the “learning loop” consistent and avoids a second outcomes table.

---

## 10) Deliverables checklist

- [ ] `db_migrations/outcomes_migration.py`
- [ ] `infra/db/models/feature_outcomes.py`
- [ ] repositories: `outcomes_repo`, `price_repo`
- [ ] domain: `domain/evaluation/*` (models/ports/calculations/dq)
- [ ] use cases: compute outcomes, backfill, aggregate performance
- [ ] Celery task: `app.interfaces.tasks.evaluation_tasks.compute_matured_outcomes`
- [ ] beat schedule entry after daily refresh/snapshot
- [ ] API endpoints: aggregates, leaderboard, ticker outcome timeline
- [ ] tests: unit tests for calculations + integration tests with sample price series

---

## Appendix: Why this is more robust & useful

- **Uses your “published daily dataset”** (Feature Store) as the canonical research corpus.
- **Produces labels once, reuses everywhere** (scans, presets, pattern analysis).
- **Avoids look‑ahead bias** by defining entry/horizons explicitly.
- **Makes the tool self‑improving**: not by guessing, but by measuring outcomes, segmenting by regime, and exposing it as a research workflow.
