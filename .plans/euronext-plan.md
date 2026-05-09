# Euronext Market Expansion Plan

## Summary
Add Euronext as market `ENX`, covering primary Euronext boards only: regulated, Growth, Access, and Expand. Defer ATHEX/Greece because Athens is linked from Euronext but not present in the same Euronext Live equity feed.

Live validation on April 29, 2026:
- Euronext “all equities” directory: `3,945` rows, `2,893` unique ISINs.
- Primary boards, all issue types: `1,832` rows, `1,822` unique ISINs.
- v1 target universe: `1,720` Common Stock listing rows, `1,710` unique ISINs.
- Breakdown: regulated `987`, Growth `559`, Access `164`, Expand `10`.

## Key Changes
- Add market identity:
  - Market: `ENX`
  - Default currency: `EUR`, but persist per-row currency from source: `EUR`, `NOK`, `GBP`, `USD`.
  - Default benchmark: Euronext 100 via Yahoo `^N100`, fallback `^FCHI`.
  - Exchange groups exposed to scan UI/API: `ENX_REGULATED`, `ENX_GROWTH`, `ENX_ACCESS`, `ENX_EXPAND`.
  - Canonical symbols use Euronext local symbol + Yahoo suffix by MIC: Paris `.PA`, Amsterdam `.AS`, Brussels `.BR`, Lisbon `.LS`, Milan `.MI`, Oslo `.OL`, Dublin `.IR`.

- Add Euronext universe ingestion:
  - Fetch from Euronext Live stock data/download endpoints linked from the Euronext product directory.
  - Include only Common Stock issue type `101`.
  - Include MIC sets for regulated, Growth, Access, and Expand; exclude Global Equity Market, EuroTLX, Trading After Hours, Expert Market, rights, warrants, preferred stock, certificates, funds, ETFs, and non-common products.
  - Store listing rows, not deduped issuers, because price/liquidity is venue-specific; record ISIN and duplicate-ISIN metadata for audit.

- Add data providers:
  - Use Euronext Live as official universe/latest-quote validation source.
  - Use `yfinance` as primary historical OHLCV and fundamentals provider for `ENX`.
  - Provider routing: `ENX -> yfinance`; field capability metadata must mark unsupported ownership/sentiment fields explicitly.
  - FX must support row-level currencies, especially `NOK`, `GBP`, and `EUR`, not only market-level default currency.

- Add classification:
  - Add `data/euronext-deep.csv`.
  - Populate `StockUniverse.sector`, `StockUniverse.industry`, and `StockIndustry` hierarchy from the best available source.
  - Use Euronext/I CB reference data if available; otherwise seed from yfinance sector/industry and mark subgroup coverage as degraded until full ICB reference data is loaded.

- Add backend/frontend/static surfaces:
  - Supported market constants, schemas, queues, runtime preferences, telemetry, provider snapshots, calendars, scan guard mappings, static export, frontend market selector, flags, symbol inference, and currency formatting.
  - Calendar handling must be exchange-aware for ENX MICs, mapping Milan to `XMIL`, Dublin to `XDUB`, Oslo to `XOSL`, and Paris/Amsterdam/Brussels/Lisbon directly.

## Launch Gates And Tests
- Universe gate:
  - Re-query Euronext Live immediately before implementation.
  - ENX Common Stock source count must match live baseline within 2%.
  - Active app universe must be at least 95% of accepted source rows unless exclusions are recorded.

- Coverage gates:
  - Latest daily OHLCV for at least 95% of active ENX symbols.
  - Market cap and core valuation fields for at least 85% overall and 95% on regulated-board symbols.
  - Sector + industry group coverage at least 95%; subgroup coverage at least 85% or launch reports degraded classification.

- Backend tests:
  - Market/currency/calendar/symbol resolver supports `ENX` and Euronext suffixes.
  - Universe adapter canonicalizes MICs, filters issue type `101`, rejects non-primary boards, preserves ISIN/source metadata, and handles dual listings deterministically.
  - Provider routing and field capability registry include `ENX`.
  - Freshness guard resolves ENX exchange groups and MIC calendars correctly.
  - FX normalization stores `EUR`, `NOK`, `GBP`, and `USD` metadata.

- Frontend/static tests:
  - ENX appears in market selector and static market pages.
  - ENX exchange groups scan correctly.
  - Symbols infer correctly from `.PA`, `.AS`, `.BR`, `.LS`, `.MI`, `.OL`, `.IR`.
  - Scan, breadth, groups, and static pages accept `ENX`.

## Assumptions And Sources
- v1 market code is `ENX`.
- v1 excludes ATHEX/Greece.
- v1 universe is primary Euronext Common Stock listings: `1,720` rows today.
- Paid Euronext Web Services/reference data is not required for initial universe and price/fundamentals, but may be required to claim full official ICB subgroup coverage.
- Sources: [Euronext equities directory](https://live.euronext.com/en/products/equities/list), [regulated](https://live.euronext.com/en/products/equities/regulated/list), [Growth](https://live.euronext.com/en/products/equities/growth/list), [Access](https://live.euronext.com/en/products/equities/access/list), [Expand](https://live.euronext.com/en/products/equities/expand/list), and [Euronext March 2026 issuer statement](https://www.euronext.com/en/about/media/euronext-press-releases/euronext-confirms-its-european-leading-position-equity-listing).
