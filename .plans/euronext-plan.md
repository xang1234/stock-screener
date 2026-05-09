# Euronext Market Expansion Plan

> **Companion**: full critique with prioritized findings, alternatives considered, and detailed Phase 0–4 sequencing lives in [`.plans/euronext-plan-critique.md`](./euronext-plan-critique.md). This file is the canonical implementation plan after folding in the 2026-05-09 decisions.

## Summary

Add Euronext as market `ENX`, covering primary Euronext boards only: regulated, Growth, Access, and Expand. Defer ATHEX/Greece because Athens is linked from Euronext but not present in the same Euronext Live equity feed. Germany is **not** part of Euronext (Frankfurt is Deutsche Börse) and follows as a separate market `DE` after `ENX` ships.

Live validation on April 29, 2026:
- Euronext "all equities" directory: `3,945` rows, `2,893` unique ISINs.
- Primary boards, all issue types: `1,832` rows, `1,822` unique ISINs.
- v1 target universe: `1,720` Common Stock listing rows, `1,710` unique ISINs.
- Breakdown: regulated `987`, Growth `559`, Access `164`, Expand `10`.

## Decisions (2026-05-09)

1. **Scope**: sequential — ship `ENX` first; `DE` follows as a separate market reusing the foundations.
2. **Germany market code**: `DE` (matches US/HK/JP/IN/CN/TW country-code convention).
3. **Calendar architecture**: extend `MarketProfile` to multi-MIC. New `calendar_ids: Mapping[str, str]` field; existing `calendar_id` retained as primary; `MarketCalendarService.last_completed_trading_day` accepts an optional `mic` parameter.
4. **Source resilience**: Playwright + stale-snapshot fallback. Build as a generic `BrowserSnapshotService` so future JS-rendered sources (LSE, SIX, etc.) reuse it.
5. **Dublin**: route Dublin lines to their `.L` LSE primary listing via ISIN cross-walk. `.IR` is dropped from `_MARKET_BY_SUFFIX`. Lines without an LSE counterpart are excluded from v1.
6. **ISIN storage**: new `stock_identifiers` linking table (PIT-correct, scheme-agnostic — supports ISIN now, FIGI/CUSIP/SEDOL later).
7. **Plan-file location**: this canonical plan lives at `.plans/euronext-plan.md`; companion critique at `.plans/euronext-plan-critique.md`.

## Key Changes

- **Market identity**:
  - Market code: `ENX`.
  - Default currency: `EUR`. Per-row currency persisted from source: `EUR`, `NOK`, `GBP`, `USD` (schema already supports per-row currency on `stock_universe.currency`).
  - Benchmark: primary `^N100` (Euronext 100); fallback `^STOXX` (broader STOXX Europe 600), not narrower CAC 40 — fallback must reduce, not concentrate, coverage risk.
  - Exchange groups exposed to scan UI/API: `ENX_REGULATED`, `ENX_GROWTH`, `ENX_ACCESS`, `ENX_EXPAND`.
  - Yahoo suffix → market mapping: Paris `.PA`, Amsterdam `.AS`, Brussels `.BR`, Lisbon `.LS`, Milan `.MI`, Oslo `.OL` map to `ENX`. **Dublin `.IR` is excluded** — Dublin lines route to their `.L` LSE counterpart via ISIN cross-walk.
  - MIC set: `XPAR, XAMS, XBRU, XLIS, XMIL, XOSL` (XDUB excluded; routed to XLON via ISIN).

- **Multi-MIC calendar architecture** (Phase 0 prerequisite):
  - Extend `MarketProfile` (`backend/app/domain/markets/registry.py`) with `calendar_ids: Mapping[str, str]` keyed by MIC. Existing markets keep their single MIC by seeding `calendar_ids = {profile.calendar_id: profile.calendar_id}` (backward-compatible).
  - `MarketCalendarService.last_completed_trading_day(market, mic=None)` resolves per-MIC when `mic` is provided; falls back to `calendar_id` (primary) otherwise.
  - `services/market_data_freshness.py::check_symbol_freshness` resolves each symbol's MIC from `StockUniverse.exchange` (or new `mic` column) and passes it.

- **Universe ingestion**:
  - Fetch from Euronext Live (regulated/Growth/Access/Expand directories) via the new generic `BrowserSnapshotService` — JS-rendered, session-cookie-gated, so a Playwright-driven scraper.
  - **Stale-snapshot fallback**: snapshots persisted to DB with `captured_at`, `is_stale BOOLEAN`. If today's fetch fails, latest non-stale snapshot is reused and tagged stale; telemetry signal `universe.stale_snapshot_used{market="ENX"}` emitted.
  - Include only Common Stock issue type `101` (see `data/governance/euronext-issue-types.md` for the code dictionary).
  - Exclude Global Equity Market, EuroTLX, Trading After Hours, Expert Market, rights, warrants, preferred stock, certificates, funds, ETFs, and non-common products.
  - Store listing rows, not deduped issuers, because price/liquidity is venue-specific; record ISIN and per-MIC metadata in the new `stock_identifiers` linking table.
  - **Dual-listing tiebreaker** (deterministic, pure function unit-tested in isolation): primary MIC = ISIN country prefix MIC if available; tiebreak by 6-month median ADV; final tiebreak alphabetical MIC.
  - **Dublin handling**: for each XDUB candidate, look up the ISIN's LSE counterpart via OpenFIGI or LSE issuer master. If present, register the symbol with `.L` suffix and `market="ENX"`, recording both `XDUB` and `XLON` in `stock_identifiers` (scheme=`MIC`). If no LSE counterpart, exclude from v1.
  - Refresh cadence: weekly via Celery beat (matches existing markets; Euronext issuer statements are monthly).
  - PIT-correct membership: soft-delete delisted rows with `delisted_on`; board promotions create new rows with `effective_from` rather than overwriting.

- **Identifiers schema** (Phase 0 prerequisite):
  - New `stock_identifiers` linking table — Alembic migration. Columns: `id PK`, `symbol`, `market`, `scheme` (`ISIN`/`FIGI`/`CUSIP`/`SEDOL`/`MIC`), `value`, `is_primary BOOLEAN`, `effective_from DATE`, `effective_to DATE NULLABLE`, `source TEXT`. Indexes on `(symbol, market)` and `(scheme, value)`.
  - Backfill ISINs for existing markets where available (HK, IN, etc. — many already in `*-deep.csv`).
  - `MarketTaxonomyService` reads/writes through this table.

- **Data providers**:
  - `yfinance` is primary for historical OHLCV and fundamentals on `ENX`.
  - Provider routing matrix `_POLICY_MATRIX` (`backend/app/services/provider_routing_policy.py`): add `"ENX": (PROVIDER_YFINANCE,)`. Bump `POLICY_VERSION`.
  - Field capability registry must mark unsupported ownership/sentiment fields explicitly.
  - **FX scope correction**: schema already supports per-row currency. Real work: extend `MARKET_CURRENCY_MAP` in `services/fx_service.py` with `EUR`, `NOK`, `GBP` (USD already present); validate Yahoo FX tickers `EURUSD=X`, `NOKUSD=X`, `GBPUSD=X`; document ECB reference rate as fallback if Yahoo FX is unavailable.

- **Classification**:
  - Add `data/euronext-deep.csv`. Columns: `Symbol, Sector, Industry, Sub-Industry, Theme, Name, Market, MIC, ISIN, Exchange-Group`.
  - Populate `StockUniverse.sector`, `StockUniverse.industry`, and `StockIndustry` hierarchy from Euronext listing pages (ICB level 1–2) + yfinance fallback.
  - Sub-industry/subgroup launches **degraded**: pre-tagged in API responses and frontend with a banner; do not delay launch waiting for paid ICB reference data. Access/Expand tier rows additionally tagged `fundamentals_quality="degraded"`.

- **Backend / frontend / static surfaces**:
  - `SUPPORTED_MARKET_CODES` (`domain/markets/market.py`): add `"ENX"`.
  - `MarketCatalogEntry` (`domain/markets/catalog.py`): add `ENX` with `calendar_ids = {"XPAR": "XPAR", "XAMS": "XAMS", "XBRU": "XBRU", "XLIS": "XLIS", "XMIL": "XMIL", "XOSL": "XOSL"}`, `currency = "EUR"`, capabilities `benchmark / breadth / fundamentals / group_rankings / feature_snapshot / official_universe = True`, `finviz_screening = False`.
  - Symbol inference `_MARKET_BY_SUFFIX` (`services/security_master_service.py`): add 6 entries (`.PA .AS .BR .LS .MI .OL`); skip `.IR`.
  - `start_celery.sh:67`: add `ENX` to case-statement allow-list. (Refactor to derive from `market_registry.supported_market_codes()` is out of scope for v1; flag as tech debt.)
  - `STATIC_EXPORT_MARKETS` in `backend/scripts/export_static_site.py`: add `"ENX"`.
  - `frontend/src/static/marketFlags.js`: add `"ENX": "🇪🇺"`. Document trade-off (Norway non-EU, future UK consideration).
  - **Per-exchange-group breadth**: extend `BreadthCalculatorService` with an `exchange_group` grouping mode in addition to `market`. Compute breadth for `ENX_REGULATED`, `ENX_GROWTH`, `ENX_ACCESS`, `ENX_EXPAND`, plus aggregate `ENX`. Aggregating 7 venues into a single read masks divergences and is misleading.

## Implementation Sequencing

Detailed Phase 0–4 sequencing is in [`euronext-plan-critique.md`](./euronext-plan-critique.md#implementation-sequencing-consequence-of-decisions-above). At a glance:

- **Phase 0 — Foundations** (touch all existing markets, backward-compatible):
  1. Multi-MIC `MarketProfile` + `MarketCalendarService` refactor.
  2. `stock_identifiers` linking table + backfill.
  3. FX service extension (EUR / NOK / GBP).
  4. Symbol inference + provider routing + supported-markets + Celery + static-export additions.
  5. Generic `BrowserSnapshotService` with stale-snapshot fallback.

- **Phase 1 — ENX universe ingestion + classification**: `enx_universe_ingestion_adapter.py`, `data/euronext-deep.csv`, `data/governance/euronext-issue-types.md`, `MarketCatalogEntry`.

- **Phase 2 — Per-group breadth + frontend**: `BreadthCalculatorService` grouping mode, market flag.

- **Phase 3 — Verification gates** (see Launch Gates And Tests below).

- **Phase 4 — DE follow-up** (separate plan file, after `ENX` ships): single-MIC `XETR`, single-currency `EUR`, ~600–650 primary equities; reuses Phase 0 foundations; benchmarks `^GDAXI` primary, `^MDAXI` fallback.

## Launch Gates And Tests

- **Universe gate**:
  - Re-query Euronext Live immediately before implementation.
  - ENX Common Stock source count matches live baseline within 2%.
  - Active app universe at least 95% of accepted source rows unless exclusions are recorded.

- **Coverage gates**:
  - Latest daily OHLCV for at least 95% of active ENX symbols.
  - Market cap and core valuation fields: 95% on regulated-board symbols (gate); ~75% on Growth (advisory); Access/Expand pre-tagged `fundamentals_quality="degraded"` and not blocking launch.
  - Sector + industry coverage at least 95% (ICB levels 1–2). Sub-industry coverage launches **degraded** — banner on API and frontend; full ICB requires paid reference data, deferred.

- **Backend tests**:
  - Multi-MIC calendar resolution: `last_completed_trading_day("ENX", mic="XOSL")` differs from `("ENX", mic="XPAR")` on a known asymmetric holiday (e.g., Bastille Day, Norwegian Constitution Day).
  - Freshness gate returns 200 for an XOSL symbol on a French holiday when Oslo prices are current.
  - Universe ingest survives a Euronext source 503: stale-snapshot fallback used, telemetry signal emitted, snapshot marked `is_stale=true`.
  - Universe adapter canonicalizes MICs, filters issue type `101`, rejects non-primary boards, preserves ISIN / source metadata in `stock_identifiers`, and handles dual listings deterministically (≥5 fixture cases including same-ISIN cross-MIC and same-symbol cross-MIC).
  - Dublin probe: at least N Dublin ISINs successfully cross-walked to `.L` and active in price cache; lines without LSE counterparts excluded.
  - Provider routing and field capability metadata include `ENX` (yfinance only).
  - FX rates table populated for EUR, NOK, GBP for launch date and 5 prior days.
  - `stock_identifiers` populated for >99% of regulated rows (ISIN scheme).

- **Frontend / static tests**:
  - ENX appears in market selector and static market pages.
  - ENX exchange groups scan correctly.
  - Symbol inference resolves `AIR.PA`, `ASML.AS`, `ABI.BR`, `EDP.LS`, `ENI.MI`, `EQNR.OL` to `ENX`.
  - Per-group breadth API returns 4 group rows + 1 aggregate row for `ENX`.
  - Static export skips closed-MIC days correctly (depends on multi-MIC calendar resolution).

## Assumptions And Sources

- v1 market code is `ENX`. Germany follows as `DE` (separate plan).
- v1 excludes ATHEX/Greece and Dublin lines without LSE counterparts.
- v1 universe is primary Euronext Common Stock listings: `1,720` rows today (less the excluded Dublin slice without LSE counterparts).
- Paid Euronext Web Services / ICE reference data is not required for initial universe and price / fundamentals, but may be required to claim full official ICB sub-industry coverage. Sub-industry launches degraded.
- yfinance fundamentals coverage is high on regulated-tier large/mid-caps and sparse on Access/Expand microcaps — pre-registered as `fundamentals_quality="degraded"`.
- Sources: [Euronext equities directory](https://live.euronext.com/en/products/equities/list), [regulated](https://live.euronext.com/en/products/equities/regulated/list), [Growth](https://live.euronext.com/en/products/equities/growth/list), [Access](https://live.euronext.com/en/products/equities/access/list), [Expand](https://live.euronext.com/en/products/equities/expand/list), and [Euronext March 2026 issuer statement](https://www.euronext.com/en/about/media/euronext-press-releases/euronext-confirms-its-european-leading-position-equity-listing).
- Cross-walk sources for Dublin → LSE: OpenFIGI API, LSE issuer master.
