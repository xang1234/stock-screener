# LL2-E1-T1 Audit: Price Ingestion and Adjustment Defects

Date: 2026-02-24  
Issue: `StockScreenClaude-ofh.1.1`  
Scope: yfinance ingestion, bulk fetch paths, cache/database write semantics, stale-data edge windows.

## 1. Audit Goal

Establish a factual baseline for raw-vs-adjusted price semantics before defining canonical contract ADR (`StockScreenClaude-ofh.1.2`).

## 2. Invariants Audited

- `INV-1`: Adjustment basis must be explicit and deterministic at every Yahoo fetch call.
- `INV-2`: `stock_prices.adj_close` must represent adjusted close, never a blind alias.
- `INV-3`: All ingestion paths must preserve one canonical OHLC basis (or persist both, with explicit factors).
- `INV-4`: End-of-day replacement must be possible for same-session intraday rows.
- `INV-5`: Read freshness checks must guard against stale intraday bars after market close.
- `INV-6`: Corporate-action artifacts (split/dividend context) must be preserved or derivable.

## 3. Ingestion Path Inventory

1. Single-symbol cache miss: `PriceCacheService.get_historical_data()` -> `_fetch_full_and_cache()` -> `YFinanceService.get_historical_data()` -> `_store_in_database()`.
2. Single-symbol incremental: `PriceCacheService._fetch_incremental_and_merge()` -> `_store_in_database()` (new rows only).
3. Bulk warmup (cache manager): `CacheManager.warm_price_cache()` -> `BulkDataFetcher.fetch_batch_data()` (`ticker.history`) -> `PriceCacheService.store_in_cache()`.
4. Bulk warmup (tasks): `cache_tasks._fetch_with_backoff()` -> `BulkDataFetcher.fetch_batch_prices()` (`yf.download`) -> `PriceCacheService.store_batch_in_cache()`.
5. Bulk miss fallback in read path: `PriceCacheService.get_many()` -> `BulkDataFetcher.fetch_batch_data()` -> `_store_in_database()`.
6. Benchmark path: `BenchmarkCacheService.get_spy_data()` -> `YFinanceService.get_historical_data()` -> `BenchmarkCacheService._store_in_database()`.

## 4. Findings (Severity Ordered)

### F1 (Critical): `adj_close` is systemically aliased to `close`

- Evidence (code):
  - `backend/app/services/price_cache_service.py:1364`
  - `backend/app/services/price_cache_service.py:1575`
  - `backend/app/services/benchmark_cache_service.py:331`
- Evidence (database):
  - `SELECT COUNT(*), SUM(adj_close = close) FROM stock_prices;` -> `10442619 | 10442572`
  - Null audit: `adj_close` and `close` null counts both `47`; non-null unequal rows `0`.
- Invariant violations:
  - Violates `INV-2` and `INV-6`.
- Impact:
  - `adj_close` column is not trustworthy as an adjusted-series contract.
  - Split/dividend-aware analytics cannot rely on persisted adjusted semantics.

### F2 (Critical): Same-day row correction is impossible (insert-only persistence)

- Evidence:
  - Existing date rows are always skipped in writes:
    - `backend/app/services/price_cache_service.py:1350`
    - `backend/app/services/price_cache_service.py:1563`
  - Incremental fetch only writes rows strictly newer than last cached date:
    - `backend/app/services/price_cache_service.py:400`
    - `backend/app/services/price_cache_service.py:438`
- Invariant violations:
  - Violates `INV-4` and contributes to `INV-5`.
- Impact:
  - If a row is first cached during market hours (partial day), later close-finalized data for that same date cannot overwrite DB.
  - This creates persistent stale/corrupted end-of-day bars in DB history.

### F3 (High): Adjustment basis is implicit (no explicit `auto_adjust` / `actions` contract)

- Evidence:
  - `ticker.history(...)` calls with default adjustment behavior:
    - `backend/app/services/yfinance_service.py:120`
    - `backend/app/services/bulk_data_fetcher.py:75`
  - `yf.download(...)` call also omits adjustment flags:
    - `backend/app/services/bulk_data_fetcher.py:433`
- Invariant violations:
  - Violates `INV-1` and `INV-3`.
- Impact:
  - Semantics can drift across yfinance versions/endpoints.
  - Raw-vs-adjusted meaning is not enforceable in code review or tests.

### F4 (High): Multiple Yahoo fetch APIs are mixed without normalization guardrails

- Evidence:
  - `ticker.history` path: `backend/app/services/bulk_data_fetcher.py:75`
  - `yf.download` path: `backend/app/services/bulk_data_fetcher.py:433`
  - Both persist through common insert-only writers.
- Invariant violations:
  - Violates `INV-3`.
- Impact:
  - If endpoint defaults differ, cache can contain mixed semantics over time.
  - No post-fetch normalization/assertions catch this.

### F5 (Medium): Intraday staleness metadata exists but is not enforced on read path

- Evidence:
  - Intraday stale detector implemented: `backend/app/services/price_cache_service.py:566`
  - Read flow freshness check uses only date threshold: `backend/app/services/price_cache_service.py:117`, `backend/app/services/price_cache_service.py:1387`
  - Detector is not called from `get_historical_data()`.
- Invariant violations:
  - Violates `INV-5`.
- Impact:
  - Data can be considered "fresh" by date while still representing intraday partial bars.

### F6 (Medium): Post-close buffer classification bug in fetch metadata

- Evidence:
  - `is_after_close` is computed but not used: `backend/app/services/price_cache_service.py:518`
  - `data_type` is set solely by `is_market_open`: `backend/app/services/price_cache_service.py:521`
- Invariant violations:
  - Weakens `INV-5`.
- Impact:
  - Fetches between market close and finalization buffer can be misclassified as closing data and skipped from refresh workflows.

### F7 (Medium): Corporate-action context is not persisted

- Evidence:
  - No split/dividend factor columns in `StockPrice` model:
    - `backend/app/models/stock.py:7`
  - No ingestion writes for action metadata in price cache services.
- Invariant violations:
  - Violates `INV-6`.
- Impact:
  - Cannot reconstruct/verify adjusted-vs-raw transitions historically from local data alone.

## 5. Edge-Window Evidence

### Split window check (NVDA 10-for-1 period)

- Sample rows around split week show continuous close series:
  - `NVDA 2024-06-07 close 120.8263`
  - `NVDA 2024-06-10 close 121.7279`
  - Ratio `2024-06-10 / 2024-06-07 = 1.00746`
- Interpretation:
  - Stored `close` behaves like an adjusted series around split date.
  - But `adj_close` is still an alias of `close`, so raw/adjusted duality is not represented.

### Stale intraday hazard

- Intraday stale handling exists as a background refresh mechanism (`get_stale_intraday_symbols`) but DB write strategy is insert-only.
- Same-session replacement is therefore blocked at persistence layer even if refresh jobs run successfully.

## 6. Affected Modules

- `backend/app/services/yfinance_service.py`
- `backend/app/services/bulk_data_fetcher.py`
- `backend/app/services/price_cache_service.py`
- `backend/app/services/benchmark_cache_service.py`
- `backend/app/tasks/cache_tasks.py`
- `backend/app/models/stock.py`

## 7. Recommended Contract Work for Next Beads

For `StockScreenClaude-ofh.1.2` (ADR):
- Define canonical fields: raw OHLC, adjusted close, adjustment factor source, and allowed derivations.
- Require explicit yfinance fetch parameters at all call sites (`auto_adjust`, action inclusion) with testable invariants.
- Specify same-day upsert semantics (replace on `(symbol,date)` for refresh windows).
- Define staleness acceptance window and read-path guard behavior.

For `StockScreenClaude-ofh.1.3` / `StockScreenClaude-ofh.1.4`:
- Implement upserts and canonical mapping in all paths.
- Reconcile historical rows where `adj_close` is aliased and corporate-action semantics are ambiguous.
