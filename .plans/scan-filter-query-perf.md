# Live vs Static Scan Comparison — Fix Preset Filtering + Investigate Relevance

## Context

The project ships **one React app in two modes** (selected at build time by `VITE_STATIC_SITE`, see `frontend/src/config/runtimeMode.js`):

- **Static** (`xang1234.github.io/stock-screener`): bakes a daily scan into chunked JSON, hydrates ~9.9k rows in the browser, then filters/sorts **client-side** (`frontend/src/static/scanClient.js`). No backend.
- **Live** (Tailscale host): runs real scans, filters **server-side in SQL**.

The user reports the live page's results look more relevant **but its presets "don't filter well."** This plan fixes the filtering defect and separately investigates result relevance. (Frontend resilience work is de-scoped to a follow-up bead — see Workstream 4.)

## What's actually broken (empirically confirmed on the live site)

Selecting a preset (e.g. *Minervini Trend Template*) works correctly on the **frontend**: all four filters become chips (`Stage: 2`, `Minervini: ≥70`, `RS: ≥70`, `MA Align: Yes`) and the request URL is built right:
`/api/v1/scans/{id}/results?...&stage=2&min_score=70&min_rs=70&ma_alignment=true`.

**But every filtered request fails with `net::ERR_ABORTED`** and the table keeps showing the stale, unfiltered rows with a cosmetic "(filtered)" label. Root cause chain:

1. The filtered results query is **catastrophically slow** — measured live (read-only timing via the authenticated page console):

   | Market | Backend path | Unfiltered | Filtered (`rs≥80 & stage=2`) | Slowdown |
   |---|---|---|---|---|
   | HK (2,770 rows) | `scan_results` table | **83 ms** | 25.6 s → 63 rows | **~300×** |
   | JP (3,742 rows) | `scan_results` table | **120 ms** | 37.3 s → 65 rows | **~310×** |
   | US (9,867 rows) | `feature_store` (`stock_feature_daily`) | 5.0 s | >90 s (never returns) | — |

   Slowdown scales with table *heaviness*, not match count (HK matches only 63 rows yet takes 25 s). Filtering is **logically correct** — it returns the right rows when allowed to finish.

2. Axios timeout is **30 s** (`frontend/src/api/client.js:9`; only `/v1/themes` gets 300 s). `getScanResults` (`frontend/src/api/scans.js:92`) does **not** forward React Query's `AbortSignal`, so the only thing that can abort the XHR is axios's own 30 s timeout → `xhr.abort()` → `ERR_ABORTED` + the "API No Response" console error.

3. The aborted request never updates the table, so the previous unfiltered page stays on screen → **"presets don't filter."** The static site is immune because it filters already-hydrated rows in JS — no backend query in the loop.

### Why it's slow — two *different* causes per path

There are **two backend read paths** (`use_cases/scanning/get_scan_results.py` routes to feature-store when a scan is bound to a feature run, else the legacy table):

**A) Legacy `scan_results` table (HK/JP, and older scans).** Filter columns are **already indexed** — composite `(scan_id, <col>)` indexes exist for nearly every field (`idx_scan_result_score`, `idx_scan_rs_*`, `idx_scan_eps_*`, `idx_scan_perf_*`, `idx_scan_beta*`; `stage`/`rs_rating` have `index=True`; plus `20260426_0017_add_scan_readiness_indexes.py`). So the 25–37 s is **not** missing indexes. The suspect is the **`count()` over a heavy SELECT**: `_scan_results_query` (`scan_result_repo.py:43`) selects the whole `ScanResult` row (large `details` JSON + `rs_sparkline_data`/`price_sparkline_data`) plus **two unconditional outer joins**, and `apply_sort_and_paginate` runs `total = query.count()` (`scan_result_query.py:203`), which SQLAlchemy wraps as `SELECT count(*) FROM (<that heavy double-join SELECT>)`. Unfiltered the planner short-circuits; add a `WHERE` and it falls back to a full scan reading every row's blobs.

**B) Feature-store `stock_feature_daily` (US daily).** Fundamentally different and worse: only `composite_score`, `overall_rating`, `as_of_date` are real indexed columns. **`stage`, `rs_rating`, `minervini_score`, `ma_alignment` — and almost every preset filter field — live inside the single `details_json` column** (confirmed: `feature_store_query.py:47` `_JSON_FIELD_MAP`; model `feature_store.py:87`). So preset filtering = **JSON extraction over the whole ~10k-row run with no supporting index**, on top of the same `count()` + double-join overhead (`feature_store_query.py:201`, `feature_store_repo.py:43`). This is why US is slow even unfiltered (5 s) and never returns when filtered. A lean `count()` alone will not fix this path.

## Recommended approach

### Workstream 0 — Instrument first (must precede any code change)

- Confirm the live DB engine **without exposing credentials**: read `engine.dialect.name` (or a sanitized URL render via `engine.url.render_as_string(hide_password=True)`) — **do not echo `$DATABASE_URL`**. Postgres vs SQLite changes the indexing options in Workstream 1B.
- Capture `EXPLAIN ANALYZE` for one filtered query on **each** path (legacy + feature-store), unfiltered vs `+rs_rating>=80,stage=2`, to confirm whether cost is the `count()` subquery, the outer joins, or JSON extraction before writing code.

### Workstream 1 — Backend query speed (primary fix)

**1A — Two-phase query for BOTH paths (replaces "conditional joins").** The joined columns (`company_name`, `market`, `exchange`, `currency`, `market_cap_usd`, `adv_usd`, ownership/sentiment, growth cadence) feed the **response**, not just filters (`scan_result_repo.py:43` / `_unpack_joined_row`; `feature_store_repo.py:43` / `_unpack_feature_joined_row`), so the joins can't simply be dropped. Instead split into two queries:
  1. **Selection/count query** — lean: select only the key (`ScanResult.id` / `(run_id, symbol)`), apply filters + sort, run `count()` and fetch the page's keys. Add the `StockUniverse`/`StockFundamental` joins here **only when** the active `FilterSpec`/`SortSpec` references a joined column.
  2. **Hydration query** — join + project full data for **only the ~50 keys** on the requested page.

This makes `count()` cheap on the legacy path and bounds the join/projection cost to one page on both paths.

**1B — Feature-store field access (the extra fix for US daily).** A lean count helps but JSON extraction stays the bottleneck. After confirming the engine (Workstream 0), choose:
  - **Preferred:** promote the hot preset filter/sort fields (`stage`, `rs_rating`, `minervini_score`, `canslim_score`, `eps_growth_qq`, `rs_rating_1m/3m/12m`, etc.) from `details_json` to **real indexed columns** on `stock_feature_daily` (populate during feature-run write; migration + backfill), or
  - **If staying JSON (Postgres only):** add **expression / generated-column indexes** on the `details_json` paths used by presets.
  - Keep `details_json` for display/SE fields. `ma_alignment` and SE-pattern booleans are lower-frequency; index only if EXPLAIN shows them dominating.

**1C — Defer heavy columns (legacy path only).** On `scan_results`, sparklines are separate columns — use `defer()`/`load_only` so `rs_sparkline_data`/`price_sparkline_data` aren't loaded when `include_sparklines` is false. **This does NOT help feature-store**, where sparklines live inside `details_json` alongside most displayed fields (`feature_store_repo.py:728`); skip it there unless that schema gains lighter projected columns.

Target: filtered preset queries return well under the 30 s axios timeout — goal <1 s on `scan_results`, low-seconds on feature-store.

### Workstream 2 — Investigate result relevance

Findings on the US auto scan (`63a38bdd…`, summary "21/9873"):
- **Default view = the entire scanned universe** (9,867 rows), sorted by score. Sorted by `minervini_score` desc, the top row is CLAR at **9.97 / stage 4** — most rows have low/absent scores; the universe dump dominates.
- `passes_only=true` → **1 row** (SENEA: composite 75.4, Minervini 81.6, CANSLIM 81.8, RS 88, stage 2 — genuinely strong).
- Two conflicting "pass" definitions: scan summary says **21 passed** (screener pass criteria) vs **1** from `passes_only` (rating ∈ {Strong Buy, Buy}, `get_scan_results.py:66`).

Investigate (analysis first, propose change separately):
- Whether the live results view should **default to `passes_only` (or a score floor)** instead of the full universe — likely the core of "static feels curated, live feels noisy" (static highlights ~2,434 charts of 9,940 rows).
- Reconcile the **two pass definitions** (21 vs 1) so the count next to a scan matches the results view.
- Spot-check default sort (composite_score) top-N per market vs the static daily run.

### Workstream 3 — HK & JP verification (DONE)

Reproduced empirically (table above): HK 83 ms→25.6 s, JP 120 ms→37.3 s. Defect is market-independent (shared legacy path); Workstream 1A/1C covers it. US additionally needs 1B.

### Workstream 4 — Frontend follow-up bead (file, even though de-scoped)

Create a bead for results-query abort/stale handling so this symptom can't silently return with the next slow filter:
- `getScanResults` doesn't accept/forward an abort signal (`frontend/src/api/scans.js:92`).
- React Query keeps stale rows visible and the UI labels them "(filtered)" (`frontend/src/features/scan/pages/ScanPageContainer.jsx:373`).
- Scope: forward `AbortSignal`, surface a real loading/error state on timeout, consider a higher results timeout. (Implementation deferred; just file it now.)

## Critical files

- `backend/app/infra/query/scan_result_query.py` — `apply_sort_and_paginate`/`apply_sort_all` (`count()`), filter application (legacy path).
- `backend/app/infra/db/repositories/scan_result_repo.py` — `_scan_results_query` (`:43`), `_scan_results_symbol_query` (`:68`), `query` (`:568`).
- `backend/app/infra/query/feature_store_query.py` — `_COLUMN_MAP`/`_JSON_FIELD_MAP` (`:29`/`:47`), `count()` (`:201`).
- `backend/app/infra/db/repositories/feature_store_repo.py` — `_feature_results_query` (`:43`), row hydration (`:728`).
- `backend/app/infra/db/models/feature_store.py` — `StockFeatureDaily` columns (`:87`); target of any field-promotion migration.
- `backend/app/use_cases/scanning/get_scan_results.py` — path routing + `passes_only` rule (`:66`); relevance/default decisions.
- `backend/app/infra/db/portability.py` — engine detection (sanitized) + JSON helpers.
- (Relevance/default view, if changed) `backend/app/api/v1/scans.py`; `frontend/src/features/scan/pages/ScanPageContainer.jsx`.

## Verification

1. **Baseline (exact, per preset × market).** Before changes, from the authenticated live page console, record the **exact** filtered result `total` for a fixed set — e.g. *Minervini*, *CANSLIM*, *Cup with Handle* (JSON/SE filter) — on US, HK, JP. (Don't conflate with scan pass-summary counts.) Capture wall-clock too.
2. **After Workstream 1.** Re-run the same calls; assert each returns the **same total** as baseline (correctness preserved) and now well under 30 s (sub-second on `scan_results`). Add a backend pytest asserting the selection/count query emits no joins and no `details`/sparkline projection when no joined column is filtered.
3. **End-to-end on live.** Load a previous US scan, apply *Minervini*; confirm the network request returns `200` (not `ERR_ABORTED`) and "Results: N (filtered)" shows the reduced exact N from step 1, not 9,867. Repeat for one HK + one JP scan and the *Cup with Handle* preset (exercises a `details_json` filter on both paths).
4. **Regression.** `make gates` (backend; includes golden) + `npm run test:run` (frontend, Node 22 via nvm). Run the new migration + backfill (1B) on a copy first; verify feature-store reads still hydrate all display fields.

> Note: per project convention (user memory), once approved I'll relocate this plan to `StockScreenClaude/.plans/` and track the work (incl. the Workstream 4 bead) in beads before implementing.
