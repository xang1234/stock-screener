# Critique: Euronext Market Expansion Plan (`ENX`)

**Source plan**: `/Users/admin/StockScreenClaude/.plans/euronext-plan.md`
**Reviewer scope**: alignment with existing codebase patterns, robustness, missing work, scope corrections.
**Date**: 2026-05-06

---

## Context

The plan adds Euronext as a single market `ENX` covering 7 venues (Paris `.PA`, Amsterdam `.AS`, Brussels `.BR`, Lisbon `.LS`, Milan `.MI`, Oslo `.OL`, Dublin `.IR`), ~1,720 Common Stock listings, multi-currency (EUR/NOK/GBP/USD), via yfinance.

This critique is grounded in the actual repo. Key existing facts:

- Markets registered in two tiers: `backend/app/domain/markets/market.py` (frozen `SUPPORTED_MARKET_CODES`) + `backend/app/domain/markets/registry.py` (`MarketProfile` with `calendar_id`, benchmarks) + `backend/app/domain/markets/catalog.py` (`MarketCatalogEntry` with capabilities).
- **Per-row currency: already supported** via `stock_universe.currency` (`backend/app/models/stock_universe.py:37`). FX metadata column exists on fundamentals.
- **`MarketProfile.calendar_id` is a single string** — not a map. `services/market_calendar_service.py::last_completed_trading_day(market)` returns one date per market.
- Celery queues derive from market code automatically (`backend/app/tasks/market_queues.py`), but `start_celery.sh:67` has a hardcoded `case "$MARKET_UPPER" in US|HK|IN|JP|KR|TW|CN)` allow-list.
- Universe adapters per market (`{market}_universe_ingestion_adapter.py`) with `lineage_hash` + `row_hash` and retry-only resilience (no stale fallback). Sources are stable static XLSX/CSV URLs (HKEX, JPX, TPEX, TWSE).
- Provider routing matrix `_POLICY_MATRIX` in `services/provider_routing_policy.py`.
- Symbol → market suffix map `_MARKET_BY_SUFFIX` in `services/security_master_service.py:28` — `.PA/.AS/.BR/.LS/.MI/.OL/.IR` are all absent today.
- yfinance default rate limit: 1 req/sec aggregate (`services/rate_limiter.py`, `settings.yfinance_rate_limit`). Bulk fetcher batches 25–200 symbols via `yf.download`.
- FX `MARKET_CURRENCY_MAP` covers USD/HKD/INR/JPY/KRW/TWD/CNY (`services/fx_service.py:59`). **EUR, NOK, GBP are not in the map.**
- `StockUniverse` model has **no `isin` column** today.
- Breadth filters by `StockUniverse.market == market` — single venue assumption.

---

## Critique

Organized by impact, highest first.

### CRITICAL-1 · Calendar architecture is a single-MIC bottleneck

**What.** The plan rolls 7 MICs (XPAR, XAMS, XBRU, XLIS, XMIL, XOSL, XDUB) under one market `ENX`, then says "Calendar handling must be exchange-aware." The current `MarketProfile.calendar_id` is a single string and `MarketCalendarService.last_completed_trading_day(market)` resolves one calendar per market. There is no mechanism today to resolve a *symbol*'s calendar.

**Why this matters.** Without a schema change, `ENX` will use one calendar (likely XPAR). On any day where Oslo (XOSL), Milan (XMIL), or Dublin (XDUB) closes for a local holiday but Paris is open — or vice versa — the freshness gate (`services/market_data_freshness.py:34`) will return false `409 market_data_stale` *or*, worse, silently miss real staleness for non-Paris symbols. XOSL has Norwegian holidays that XPAR doesn't (Constitution Day, etc.), and XPAR has French holidays XOSL doesn't (Bastille Day, etc.). In a typical year there are 6–10 such asymmetric trading days.

**Proposed change.**
- Extend `MarketProfile` with `calendar_ids: Mapping[str, str]` (keyed by exchange/MIC) in addition to the existing `calendar_id` (kept as the primary).
- Extend `StockUniverse` with `mic` or `exchange_mic` populated at ingestion time.
- Refactor `MarketCalendarService.last_completed_trading_day` to take an optional `mic` param; freshness gate then resolves per symbol.
- Make this prerequisite work; do not start ENX universe ingestion until calendar resolution is multi-MIC.

**Impact**: Critical. Trade-offs: meaningful refactor touching the freshness boundary; without it, ENX silently produces wrong staleness signals. **Effort**: Medium.

---

### CRITICAL-2 · Source acquisition strategy is fragile and underspecified

**What.** Plan says: "Fetch from Euronext Live stock data/download endpoints linked from the Euronext product directory." `live.euronext.com` is a JS-rendered SPA with session cookies, CSRF tokens, and POST-driven XLSX downloads. Existing market adapters (HKEX, JPX, TPEX, TWSE) point at *stable static XLSX URLs* — meaningfully simpler and more durable.

**Why this matters.** This is the difference between a 50-line `requests.get` adapter (current pattern) and a Playwright-driven scraper that breaks on minor frontend changes. The existing `OfficialMarketUniverseSourceService._http_get_with_retries` retries on connection/timeout but **has no cached fallback**: failures fail the ingestion task loudly. ENX brings sourcing risk that the codebase isn't currently designed for.

**Proposed change.**
- Investigate alternative sources before committing: Euronext FTP (issuer-files; some are public, others on request); ICE/ESMA reference data files; OpenFIGI for cross-walking; XBRL filings to ESMA's OAM.
- If no static source exists, build the adapter behind a **stale-snapshot fallback**: keep the previous successful snapshot in the DB; if today's fetch fails, the universe falls back to yesterday with a telemetry signal (`universe.stale_snapshot_used{market="ENX"}`).
- Separately, add a Playwright-driven snapshot recorder under `backend/scripts/` so a human can regenerate the snapshot on demand if the live source moves.

**Impact**: Critical. Without source resilience, an Euronext UI change will silently break daily ingest. **Effort**: Medium-Large (depends on whether a stable source exists).

---

### CRITICAL-3 · Yahoo `.IR` (Dublin) coverage is patchy and the suffix map collision is not analyzed

**What.** Plan asserts `.IR` for Dublin. The current `_MARKET_BY_SUFFIX` in `security_master_service.py` does not include `.IR`. Two real risks:

1. Many ISEQ stocks have *no* yfinance coverage at all under `.IR`; they're only available as US ADRs or via the LSE listing (`.L`). Real `.IR` coverage on yfinance is roughly 25–35 names — far less than the plan implies.
2. `.IR` collides semantically with country code IR (Iran) — not a runtime collision in this repo's logic, but a smell.

**Why this matters.** Plan claims 1,720 ENX symbols. If `.IR` covers only ~30 yfinance-resolvable names of the ~50 Dublin lines, the launch gate ("at least 95% of source rows have OHLCV") fails on the Dublin slice unless Dublin is dropped or routed to LSE-listing equivalents.

**Proposed change.**
- Run a yfinance probe for all Dublin ISIN-resolved candidates *before* implementation; record actual coverage.
- If <50%, either (a) drop XDUB from v1 (defer like ATHEX), (b) route Dublin lines to their `.L` LSE-listing where present (and document the cross-listing rule), or (c) add a secondary provider for Dublin only.
- Add `.PA/.AS/.BR/.LS/.MI/.OL/.IR` to `_MARKET_BY_SUFFIX` regardless.

**Impact**: Critical. **Effort**: Small (probe is cheap; decision is policy).

---

### HIGH-4 · "FX must support row-level currencies" overstates required work

**What.** The plan says "FX must support row-level currencies, especially `NOK`, `GBP`, and `EUR`, not only market-level default currency." The schema *already* supports this: `StockUniverse.currency` is per-row, `StockFundamental.fx_metadata` is per-row.

**Why this matters.** The plan misidentifies the scope and risk. The real work is:
- Extend `MARKET_CURRENCY_MAP` (`fx_service.py:59`) — currently 7 currencies — to add EUR, NOK, GBP. USD already present.
- Validate that `EURUSD=X`, `NOKUSD=X`, `GBPUSD=X` Yahoo FX tickers resolve and have stable history.
- Ensure ingestion populates `StockUniverse.currency` correctly per row (Oslo lines → NOK, Dublin GBP-denominated lines → GBP, etc.).
- Update FX cache TTL/scheduling for 3 new pairs.

**Proposed change.** Replace the plan's "FX must support row-level currencies" bullet with:
> *Extend FX service*: add `EUR`, `NOK`, `GBP` to `MARKET_CURRENCY_MAP`; validate Yahoo FX tickers; document fallback if Yahoo FX is unavailable (e.g., ECB reference rate).

**Impact**: High (correctly framing avoids misallocated effort). **Effort**: Small.

---

### HIGH-5 · Breadth math will be misleading across 7 venues

**What.** `BreadthCalculatorService` filters by `StockUniverse.market == self.market`. Rolling 7 venues into `ENX` collapses ~990 Paris regulated + 559 Growth + 164 Access + 10 Expand + Oslo/Milan/Dublin lines into one daily breadth read.

**Why this matters.** Paris dominates by count and float, masking divergences in Oslo (energy-heavy) or Milan (financials-heavy). For traders who use breadth as a divergence signal, this number is actively misleading at the Euronext aggregate level.

**Proposed change.** Plan already defines exchange groups: `ENX_REGULATED`, `ENX_GROWTH`, `ENX_ACCESS`, `ENX_EXPAND`. Extend breadth to compute per exchange group (and optionally per MIC) in addition to the aggregate. The aggregation primitive already exists in the breadth service — it needs a new grouping key.

**Impact**: High. **Effort**: Medium.

---

### HIGH-6 · Benchmark fallback is narrower than primary, not broader

**What.** Plan: primary `^N100` (Euronext 100, ~100 names), fallback `^FCHI` (CAC 40, 40 names). Fallback is *narrower* than primary — that's the wrong direction for a fallback.

**Why this matters.** Fallbacks should reduce coverage risk, not concentrate it. If Euronext 100 is unavailable, CAC 40 narrows the benchmark to ~40 mostly French large-caps; the resulting RS values for Lisbon, Oslo, and Dublin lines become almost meaningless. A broader fallback (STOXX Europe 600 = `^STOXX`, or `^SXXP`) reduces concentration risk.

**Proposed change.** Primary `^N100`. Fallback `^STOXX` (broader). Optional second fallback `^FCHI` only if both fail.

**Impact**: High (affects RS rating quality across the universe). **Effort**: Small.

---

### HIGH-7 · `StockUniverse.isin` column does not exist; plan stores ISIN

**What.** Plan: "record ISIN and duplicate-ISIN metadata for audit." Schema today has no `isin` column on `stock_universe` (`backend/app/models/stock_universe.py`).

**Why this matters.** This requires an Alembic migration the plan does not mention. Worse, ISIN is the *only* useful cross-walk for Euronext multi-venue listings (7 MICs share ISINs); without it, dual-listing dedupe is impossible.

**Proposed change.** Two options, prefer (b):
- (a) Add `isin VARCHAR(12)` column to `stock_universe`.
- (b) Create a separate `stock_identifiers` linking table keyed on `(symbol, market)` with rows for `(scheme, value)` — supports ISIN now, FIGI/CUSIP/SEDOL later. More flexible for future markets and PIT-correct (an ISIN can change for a single security).

Make the migration a prerequisite of the ENX ingestion adapter.

**Impact**: High (mandatory for the plan's stated goal). **Effort**: Small (migration) to Medium (linking-table design).

---

### HIGH-8 · yfinance fundamentals coverage on Access/Expand will fail the launch gate

**What.** Plan launch gate: "Market cap and core valuation fields for at least 85% overall and 95% on regulated-board symbols." yfinance fundamentals coverage on European microcaps (Access tier ~164, Expand ~10) is typically 30–50%.

**Why this matters.** Aggregating regulated (~987) + Growth (559) + Access (164) + Expand (10), the 85% overall target is plausible *only if* regulated+Growth carry it. But many Growth-tier names also have sparse yfinance fundamentals.

**Proposed change.** Pre-register the launch reality: regulated meets 95% (gate), Growth meets ~75% (advisory), Access/Expand explicitly tagged "fundamentals-degraded" in API responses and frontend with a banner. Do not delay launch waiting for paid Euronext reference data.

**Impact**: High (prevents launch failure on a definitional gate). **Effort**: Small.

---

### MEDIUM-9 · Issue type "101" is undocumented Euronext magic number

**What.** Plan: "Include only Common Stock issue type `101`." This is a Euronext-internal classifier. Nowhere in the plan is the code dictionary captured.

**Proposed change.** Add `data/governance/euronext-issue-types.md` (or `.csv`) documenting Euronext's issue-type code dictionary with each code's inclusion/exclusion decision. Reference this document from the universe adapter docstring. Same governance pattern as `data/governance/telemetry_audit/`.

**Impact**: Medium (documentation hygiene; prevents drift). **Effort**: Small.

---

### MEDIUM-10 · Universe refresh cadence is unspecified

**What.** No mention of how often the universe re-pulls. Existing markets refresh weekly via Celery beat.

**Proposed change.** Specify weekly refresh in the plan; add the Celery beat schedule entry and document. Account for Euronext's monthly issuer statement cadence — once-a-week ingestion is sufficient.

**Impact**: Medium. **Effort**: Trivial.

---

### MEDIUM-11 · Dual-listing rule is hand-waved

**What.** Plan tests for "handles dual listings deterministically" but does not state the rule.

**Proposed change.** Define explicitly: prefer the listing whose MIC matches the ISIN's country prefix (e.g., NL00... → XAMS); on tie, prefer highest 6-month median dollar volume; on still-tie, alphabetical MIC. Capture in plan and adapter docstring. Encode the rule as a pure function tested in isolation.

**Impact**: Medium. **Effort**: Small.

---

### MEDIUM-12 · Point-in-time / membership-change handling is missing

**What.** Plan says nothing about delistings, board promotions (Growth → Regulated), or IPOs over time. Backtests need PIT-correct membership.

**Proposed change.** Use the existing `lineage_hash`/`row_hash` pattern. Add to plan:
- Soft-delete delisted rows with `delisted_on` timestamp.
- Board promotions create a new row with `effective_from` rather than overwriting.
- Document survivor-bias guidance for any backtest that uses ENX universe history.

**Impact**: Medium (matters more after launch as history accumulates). **Effort**: Small.

---

### MEDIUM-13 · `start_celery.sh` market allow-list is hardcoded

**What.** `start_celery.sh:67` has `case "$MARKET_UPPER" in US|HK|IN|JP|KR|TW|CN)`. Adding ENX requires editing this *and* the registry — duplicated source of truth.

**Proposed change.** Either (cheap) add `ENX` to the case statement, or (cleaner) refactor `start_celery.sh` to derive the allow-list from `market_registry.supported_market_codes()` via a helper Python script. Cheap option is fine for v1; flag the technical debt.

**Impact**: Medium. **Effort**: Small.

---

### LOW-14 · Symbol regex must be more permissive than HK/JP/TW

Euronext local symbols are 1–12 char alphanumeric with occasional dots/dashes (e.g., `AC.PA`, `AIR.PA`, `ALSCH.PA`, `ABBN.MI`). The HK 4-padded-digit and JP 3–5-digit regexes won't transfer. Adapter regex needs to be looser. Effort: Trivial. Flag, don't gate.

---

### LOW-15 · Frontend market flag — `🇪🇺` is fine but document it

Norway (NOK) is non-EU; UK (GBP) is non-EU; Ireland (EUR) is EU. The flag is decorative; pick `🇪🇺` and document the trade-off in `frontend/src/static/marketFlags.js`. Effort: Trivial.

---

### LOW-16 · Static export and current branch interaction

The current branch `feat/static-site-skip-closed-market` is doing calendar-aware static export work. ENX's multi-MIC nature magnifies the closed-market problem. Sequence the work so multi-MIC calendar resolution (CRITICAL-1) is in place *before* the static-export piece is finalized for ENX. Effort: Sequencing only.

---

## Summary table

| # | Change | Impact | Effort | Priority |
|---|---|---|---|---|
| 1 | Multi-MIC calendar resolution in `MarketProfile` + freshness | Critical | Medium | P0 |
| 2 | Source-acquisition resilience (alt source + stale fallback) | Critical | Medium-Large | P0 |
| 3 | Probe `.IR` Dublin yfinance coverage; decide drop/route | Critical | Small | P0 |
| 4 | Reframe "FX row-level currencies" to FX-service extension | High | Small | P1 |
| 5 | Per-exchange-group breadth (`ENX_REGULATED` etc.) | High | Medium | P1 |
| 6 | Benchmark fallback to `^STOXX` (broader), not `^FCHI` | High | Small | P1 |
| 7 | ISIN storage (linking table preferred) | High | Small-Medium | P1 |
| 8 | Pre-register Access/Expand fundamentals-degraded launch | High | Small | P1 |
| 9 | Document Euronext issue-type dictionary | Medium | Small | P2 |
| 10 | Specify weekly universe refresh cadence | Medium | Trivial | P2 |
| 11 | Define dual-listing tiebreaker rule | Medium | Small | P2 |
| 12 | PIT membership-change handling | Medium | Small | P2 |
| 13 | Decide on `start_celery.sh` allow-list (edit vs refactor) | Medium | Small | P3 |
| 14 | Loosen Euronext symbol regex | Low | Trivial | P3 |
| 15 | Document `🇪🇺` flag trade-off | Low | Trivial | P3 |
| 16 | Sequence after `feat/static-site-skip-closed-market` | Low | Sequencing | P3 |

---

## Critical files referenced

- `backend/app/domain/markets/market.py` — `SUPPORTED_MARKET_CODES`
- `backend/app/domain/markets/registry.py` — `MarketProfile` (needs `calendar_ids` map)
- `backend/app/domain/markets/catalog.py` — `MarketCatalogEntry`, `MarketCapabilities`
- `backend/app/services/market_calendar_service.py` — `last_completed_trading_day`, multi-MIC refactor target
- `backend/app/services/market_data_freshness.py` — `check_symbol_freshness`, freshness gate
- `backend/app/services/security_master_service.py` — `_MARKET_BY_SUFFIX`
- `backend/app/services/fx_service.py` — `MARKET_CURRENCY_MAP` extension
- `backend/app/services/breadth_calculator_service.py` — per-group breadth target
- `backend/app/services/provider_routing_policy.py` — `_POLICY_MATRIX` add `"ENX": (PROVIDER_YFINANCE,)`
- `backend/app/services/official_market_universe_source_service.py` — adapter base; needs ENX fetcher
- `backend/app/services/{hk,jp,tw}_universe_ingestion_adapter.py` — pattern for `enx_universe_ingestion_adapter.py`
- `backend/app/models/stock_universe.py` — needs ISIN storage decision
- `backend/app/tasks/market_queues.py` — automatic queue derivation (no change)
- `backend/app/tasks/universe_tasks.py` — dispatcher (`if snapshot.market == "ENX"` branch)
- `backend/scripts/export_static_site.py` — `STATIC_EXPORT_MARKETS`
- `frontend/src/static/marketFlags.js` — flag entry
- `start_celery.sh:67` — case statement allow-list
- `data/euronext-deep.csv` (new) — taxonomy CSV
- `data/governance/euronext-issue-types.md` (new) — code dictionary

---

## Verification checklist (additions to plan's existing gates)

- [ ] `last_completed_trading_day("ENX", mic="XOSL")` returns the correct date on a day Oslo trades but Paris is closed (and vice versa).
- [ ] Freshness gate returns 200 for an XOSL symbol on a French holiday when Oslo prices are current.
- [ ] Universe ingest survives an Euronext source 503 with stale-snapshot fallback and emits the telemetry signal.
- [ ] Yahoo coverage probe report attached to the launch ticket per MIC.
- [ ] Migration applied: ISIN storage (column or table) populated for >99% of regulated rows.
- [ ] FX rates for EUR, NOK, GBP populate `fx_rate` table for the launch date and 5 prior days.
- [ ] Breadth API returns per-exchange-group rows for ENX (`ENX_REGULATED`, `ENX_GROWTH`, `ENX_ACCESS`, `ENX_EXPAND`) and the aggregate.
- [ ] Symbol inference: probe `AIR.PA`, `ASML.AS`, `ABI.BR`, `EDP.LS`, `ENI.MI`, `EQNR.OL`, `KYG.IR` resolve to `ENX`.
- [ ] Dual-listing rule unit-tested with at least 5 fixture cases including same-ISIN cross-MIC and same-symbol cross-MIC.
- [ ] Static export skips closed MIC days correctly (depends on CRITICAL-1).

---

## Scope expansion: Germany (added 2026-05-09)

User has indicated Germany is also in scope for European expansion.

### Germany is not part of Euronext

Frankfurt is **Deutsche Börse**, not Euronext. Treating Germany under `ENX` is incorrect; it must be a separate market code (proposed: `DE` or `XETR`).

| Group | MICs | Yahoo suffixes | Currency |
|---|---|---|---|
| Euronext (`ENX`) | XPAR, XAMS, XBRU, XLIS, XMIL, XOSL, XDUB | `.PA .AS .BR .LS .MI .OL .IR` | EUR / NOK / GBP / USD |
| Deutsche Börse (`DE`) | XETR (primary), XFRA (floor) | `.DE` (XETR), `.F` (Frankfurt) | EUR |

### Counts and availability

**Euronext (live snapshot from plan, 2026-04-29):**
- Full directory: 3,945 listings / 2,893 unique ISINs
- Primary boards all issue types: 1,832 / 1,822
- v1 Common Stock target: **1,720 / 1,710** — regulated 987, Growth 559, Access 164, Expand 10
- Approx per-MIC: Paris ~750, Milan ~400, Amsterdam ~150, Brussels ~120, Oslo ~200, Lisbon ~40, Dublin ~30–40

**Germany (estimate; verify before commit):**
- XETR Prime Standard ~330, General Standard ~250, Scale ~50 → **~600–650 primary equities**
- Full Frankfurt directory inflated by foreign cross-listings; ignore the noise tier.
- Probe needed before committing exact count.

### Universe / classification availability

| Market | Universe source | Sector + Industry (ICB 1–2) | Sub-industry (ICB 3–4) |
|---|---|---|---|
| `ENX` | live.euronext.com (JS-rendered, fragile); fallback alts: OpenFIGI, ESMA OAM, paid Euronext Web Services / ICE | Scrapeable from Euronext listing pages + yfinance fallback | Degraded without paid ICB feed; yfinance approximation only |
| `DE` | boerse-frankfurt.de segment lists; deutsche-boerse.com issuer master | Scrapeable + yfinance; high coverage on DAX/MDAX/SDAX | Same degraded constraint (paid ICB or GICS) |

Both markets: sector+industry (top 2 levels) is achievable with the existing `*-deep.csv` pattern. Sub-industry launches degraded unless reference data is licensed — already in the original plan as an explicit gate.

### Scope option

- **A — Sequential (recommended)**: ship `ENX` first, then add `DE` as a separate market. Lower risk; multi-MIC + multi-currency lessons from `ENX` inform `DE`.
- **B — Parallel**: design `ENX` and `DE` together as one "European expansion" workstream. Better economy of scale (FX EUR done once; multi-market static export done once) but doubles launch surface.

If parallel, sequence the *foundation* work first (multi-MIC calendar resolution, FX extension to EUR/NOK/GBP, ISIN storage, Yahoo suffix map extension to all 9 suffixes), then ingestion adapters in parallel per market.

### New ENX-vs-DE differences worth noting

- `DE` is **single-currency (EUR)** and **single-MIC (XETR primary)** — far simpler than `ENX`. None of CRITICAL-1's multi-MIC pain applies.
- `DE` source acquisition (boerse-frankfurt.de or Xetra reference) is reportedly more stable than live.euronext.com — CRITICAL-2 risk is lower.
- `DE` benchmark: `^GDAXI` (DAX 40) primary, `^MDAXI` (MDAX) fallback for mid-cap coverage. `^SDAXI` (SDAX) optionally.
- `DE` stocks have rich yfinance fundamentals coverage on Prime/General Standard (high quality); Scale microcaps weaker.

---

## Decisions (2026-05-09)

User confirmed:

1. **Scope**: Option A — sequential. Ship `ENX` first; `DE` follows as a separate market reusing the foundations.
2. **Germany market code**: `DE` (matches US/HK/JP/IN/CN/TW country-code convention).
3. **Calendar architecture**: extend `MarketProfile` to multi-MIC. New `calendar_ids: Mapping[str, str]` field; existing `calendar_id` retained as primary; `MarketCalendarService.last_completed_trading_day` accepts an optional `mic` parameter.
4. **Source resilience**: Playwright + stale-snapshot fallback. Build as a generic `BrowserSnapshotService` so future JS-rendered sources (LSE, SIX) reuse it.
5. **Dublin**: route Dublin lines to their `.L` LSE primary listing via ISIN cross-walk. `.IR` is dropped from `_MARKET_BY_SUFFIX`. Lines without an LSE counterpart are excluded from v1.
6. **ISIN storage**: new `stock_identifiers` linking table (PIT-correct, scheme-agnostic — supports ISIN now, FIGI/CUSIP/SEDOL later).
7. **Plan-file location**: mirror this critique to `/Users/admin/StockScreenClaude/.plans/euronext-plan-critique.md` once plan mode exits.

---

## Implementation sequencing (consequence of decisions above)

### Phase 0 · Foundations (prerequisites for `ENX` and reused by `DE`)

These changes touch all existing markets but are backward-compatible.

1. **Multi-MIC `MarketProfile`** — `backend/app/domain/markets/registry.py`
   - Add `calendar_ids: Mapping[str, str] = field(default_factory=dict)` (keyed by MIC code, value = exchange_calendars/pmcal id).
   - Existing markets seed `calendar_ids = {profile.calendar_id: profile.calendar_id}` so behavior is unchanged.
   - `MarketCalendarService.last_completed_trading_day(market, mic=None)` resolves: `mic` → `calendar_ids[mic]` → calendar; if `mic` is None, falls back to `calendar_id` (primary).
   - Update `services/market_data_freshness.py::check_symbol_freshness` to look up each symbol's MIC from `StockUniverse.exchange` (or new `mic` column) and pass it.

2. **`stock_identifiers` linking table** — new Alembic migration
   - Columns: `id PK`, `symbol`, `market`, `scheme` (`ISIN`/`FIGI`/`CUSIP`/`SEDOL`/`MIC`), `value`, `is_primary BOOLEAN`, `effective_from DATE`, `effective_to DATE NULLABLE`, `source TEXT`, indexes on `(symbol, market)` and `(scheme, value)`.
   - Backfill ISINs for existing markets where available (HK, IN, etc. — many already in `*-deep.csv`).
   - Update `MarketTaxonomyService` to read/write through this table.

3. **FX service extension** — `backend/app/services/fx_service.py`
   - Extend `MARKET_CURRENCY_MAP` with `"ENX": "EUR"` and supplementary supported currencies set: `EUR, NOK, GBP, USD`.
   - Validate Yahoo FX tickers `EURUSD=X`, `NOKUSD=X`, `GBPUSD=X` resolve and have stable history.
   - Cache TTL/scheduling for new pairs.
   - Document ECB reference rate as fallback if Yahoo FX is unavailable.

4. **Symbol inference + provider routing** — small additions
   - `_MARKET_BY_SUFFIX` (`backend/app/services/security_master_service.py:28`): add `(".PA", "ENX"), (".AS", "ENX"), (".BR", "ENX"), (".LS", "ENX"), (".MI", "ENX"), (".OL", "ENX")`. **Skip `.IR`** per decision 5.
   - `_POLICY_MATRIX` (`backend/app/services/provider_routing_policy.py`): add `"ENX": (PROVIDER_YFINANCE,)`. Bump `POLICY_VERSION`.
   - `SUPPORTED_MARKET_CODES` (`backend/app/domain/markets/market.py`): add `"ENX"`.
   - `start_celery.sh:67`: add `ENX` to case statement.
   - `STATIC_EXPORT_MARKETS` (`backend/scripts/export_static_site.py`): add `"ENX"`.

5. **Generic `BrowserSnapshotService`** — new service
   - Wraps Playwright with retry, cookie reuse, and on-disk artifact capture.
   - Generic enough to be parameterized by URL pattern + post-load action (click "download" button, intercept XHR, etc.).
   - Reusable for future markets with JS-rendered sources.
   - Stale-snapshot fallback: snapshots persisted to DB with `captured_at`, `is_stale BOOLEAN`. If fresh fetch fails, the latest non-stale snapshot is reused and tagged `is_stale=true`; telemetry signal `universe.stale_snapshot_used{market="ENX"}` emitted.

### Phase 1 · ENX universe ingestion + classification

6. **`enx_universe_ingestion_adapter.py`** — new file mirroring `hk_universe_ingestion_adapter.py` pattern
   - Per-MIC parsing: XPAR, XAMS, XBRU, XLIS, XMIL, XOSL, XDUB.
   - Issue type filter: only Euronext code `101` (Common Stock).
   - Board filter: regulated, Growth, Access, Expand.
   - Dual-listing tiebreaker (deterministic): primary MIC = ISIN country prefix MIC if available; tiebreak by 6-month median ADV; final tiebreak alphabetical MIC.
   - **Dublin handling**: for each Dublin candidate (XDUB), look up the ISIN's LSE counterpart via OpenFIGI or LSE issuer master. If present, register the symbol with `.L` suffix and `market="ENX"` but record `XDUB` and `XLON` both in `stock_identifiers` (scheme=`MIC`). If no LSE counterpart, exclude.
   - Emits `CanonicalEnxUniverseRow` with `lineage_hash` + `row_hash`.

7. **`data/euronext-deep.csv`** — taxonomy CSV
   - Columns: `Symbol, Sector, Industry, Sub-Industry, Theme, Name, Market, MIC, ISIN, Exchange-Group`.
   - Sector + Industry seeded from Euronext listing pages (ICB level 1–2) + yfinance fallback.
   - Sub-Industry left sparse where ICB feed unavailable; loader handles `None` (existing pattern).

8. **`data/governance/euronext-issue-types.md`** — code dictionary documenting Euronext issue-type codes with inclusion/exclusion decision.

9. **`MarketCatalogEntry` for ENX** — `backend/app/domain/markets/catalog.py`
   - `calendar_ids = {"XPAR": "XPAR", "XAMS": "XAMS", "XBRU": "XBRU", "XLIS": "XLIS", "XMIL": "XMIL", "XOSL": "XOSL"}` (XDUB excluded; Dublin → `.L` routes through XLON, optionally registered when DE/UK markets land).
   - `currency = "EUR"`, supplementary per-row currencies allowed via `stock_universe.currency`.
   - Capabilities: `benchmark=True`, `breadth=True` (per-group), `fundamentals=True` (with Access/Expand degraded), `group_rankings=True`, `feature_snapshot=True`, `official_universe=True`, `finviz_screening=False`.
   - Benchmarks: primary `^N100`, fallback `^STOXX` (broader, not narrower).

### Phase 2 · Per-group breadth + frontend

10. **Per-exchange-group breadth** — `BreadthCalculatorService`
    - Add an `exchange_group` grouping mode in addition to `market`.
    - Compute breadth for `ENX_REGULATED`, `ENX_GROWTH`, `ENX_ACCESS`, `ENX_EXPAND`, plus aggregate `ENX`.

11. **Frontend market flag** — `frontend/src/static/marketFlags.js` add `"ENX": "🇪🇺"`.

12. **Frontend market selector + capabilities** — automatic from `MarketCatalog.as_runtime_payload()` once entry is added (no manual frontend wiring needed beyond the flag).

### Phase 3 · Verification gates

13. Pre-launch verification (in addition to plan's existing gates):
    - Multi-MIC calendar resolution: `last_completed_trading_day("ENX", mic="XOSL")` differs from `("ENX", mic="XPAR")` on a known asymmetric holiday.
    - Stale-snapshot fallback emits telemetry on simulated source 503.
    - `stock_identifiers` populated for >99% of regulated rows (ISIN scheme).
    - FX rates table populated for EUR, NOK, GBP for launch date and 5 prior days.
    - Per-group breadth API returns 4 group rows + 1 aggregate row.
    - Symbol inference probe: `AIR.PA, ASML.AS, ABI.BR, EDP.LS, ENI.MI, EQNR.OL` resolve to `ENX`.
    - Dublin probe: at least N Dublin ISINs successfully cross-walked to `.L` and active in price cache.
    - Coverage gates from original plan unchanged but Access/Expand pre-tagged as `fundamentals_quality="degraded"` in the universe row.

### Phase 4 · `DE` follow-up (deferred, separate plan file)

After `ENX` is in production:
- `de_universe_ingestion_adapter.py` from boerse-frankfurt.de or Xetra reference data.
- Single-MIC (`XETR`), single-currency (`EUR`) → much simpler than `ENX`. Reuses Phase 0 foundations.
- Benchmarks: primary `^GDAXI` (DAX 40), fallback `^MDAXI` (MDAX, 50 mid-caps for broader coverage).
- ~600–650 primary equities expected; probe to confirm before launch.
- Reuses `BrowserSnapshotService` if a Playwright-driven source is needed.

---

## Post-exit action

This critique lives at `~/.claude/plans/review-and-critique-the-idempotent-dewdrop.md` per plan-mode constraints. Will mirror to `/Users/admin/StockScreenClaude/.plans/euronext-plan-critique.md` immediately after plan-mode exit, and offer to fold accepted revisions back into `.plans/euronext-plan.md` as the canonical plan.

---

## Note on plan-file location

The user memory `feedback_plan_location.md` says project plans/design docs live in `/Users/admin/StockScreenClaude/.plans/<kebab-name>.md`, version-controlled and reviewable in PRs. Plan mode pinned this critique to `~/.claude/plans/` per its hard constraint. After exiting plan mode, recommend mirroring this to `.plans/euronext-plan-critique.md` (and ideally folding the accepted revisions back into `.plans/euronext-plan.md`).
