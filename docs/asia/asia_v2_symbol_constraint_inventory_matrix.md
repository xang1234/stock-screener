# ASIA v2 Symbol Constraint Inventory Matrix (ST1)

- Date: 2026-04-11
- Status: Completed for `StockScreenClaude-asia.2.1.1`
- Goal: Enumerate ticker/symbol constraints and assumptions across DB, ORM, API, validators, tests, and frontend usage.

## Why This Exists

ASIA expansion requires multi-market symbols and local code forms (`0700.HK`, `7203.T`, `2330.TW`, `3008.TWO`).
A safe migration needs explicit inventory of all symbol-width and format assumptions before schema/model edits.

## Summary Risk Statement

Primary blockers are not a single regex but **distributed assumptions**:

1. Multiple core tables still use `String(10)` for symbols.
2. Some validators only allow US-style uppercase 1-5 tickers.
3. Legacy universe contracts are US-enum scoped (NYSE/NASDAQ/AMEX, SP500).
4. One API path hard-rejects symbol length above 10.

## Inventory Matrix

| Layer | Location | Current Constraint | ASIA Risk | Migration Action |
|---|---|---|---|---|
| DB/ORM | `backend/app/models/stock.py` (`StockPrice.symbol`, `StockFundamental.symbol`, `StockTechnical.symbol`, `StockIndustry.symbol`) | `String(10)` | Insufficient margin for long suffixed/cross-venue IDs | Expand width using unified symbol policy in ST2 |
| DB/ORM | `backend/app/models/stock_universe.py` (`StockUniverse.symbol`, `StockUniverseStatusEvent.symbol`) | `String(10)` | Universe ingest can truncate/reject longer canonical IDs | Expand and backfill audit compatibility |
| DB/ORM | `backend/app/models/scan_result.py` (`ScanResult.symbol`) | `String(10)` | Scan output persistence incompatible with extended symbols | Expand width and reindex in migration |
| DB/ORM | `backend/app/models/provider_snapshot.py` (`ProviderSnapshotRow.symbol`) | `String(10)` | Snapshot normalization cannot preserve long market IDs | Expand width before non-US provider cutover |
| DB/ORM | `backend/app/models/theme.py` (`ThemeConstituent.symbol`) | `String(10)` | Theme constituents may drop valid non-US symbols | Expand width and update validators |
| DB/ORM | `backend/app/models/watchlist.py` (`WatchlistItem.symbol`) | `String(10)` | Watchlist add/import can reject valid suffixed symbols | Expand width |
| DB/ORM | `backend/app/models/industry.py` (`IBDIndustryGroup.symbol`, `top_symbol`) | `String(10)` | Group-rank paths assume short US symbols | Expand or isolate US-only semantics explicitly |
| DB/ORM (less strict) | `backend/app/models/user_watchlist.py`, `user_theme.py`, `ticker_validation.py`, `institutional_ownership.py` | `String(20)` | Better headroom, but inconsistent with `String(10)` core tables | Keep/align to canonical width policy |
| Validator | `backend/app/services/theme_extraction_service.py` (`ticker_pattern = ^[A-Z]{1,5}$`) | US-only ticker regex | Rejects `.HK/.TW/.TWO` and numeric-prefixed symbols | Replace with market-aware validator via SecurityMaster + active universe |
| Validator | `backend/app/services/watchlist_import_service.py` (`^[A-Z][A-Z0-9.\-]{0,9}$`) | max length 10, first char alpha | Rejects numeric-leading symbols like `0700.HK` | Update pattern + normalize through resolver |
| API validation | `backend/app/api/v1/fundamentals.py` | Rejects `len(symbol) > 10` | Blocks non-US longer symbols at API edge | Replace hard length cap with canonical symbol validator |
| Contract | `backend/app/schemas/universe.py` (`Exchange`, `IndexName`) | Exchange enum limited to NYSE/NASDAQ/AMEX; index enum only SP500 | Market universe contract cannot encode HK/JP/TW directly | Extend enums + add MARKET-style typed routing in E2.3 |
| Contract | `backend/app/schemas/scanning.py` + `frontend/src/api/scans.js` docs | Legacy universe values mention US-only options | UI/API contracts imply US-only scope | Update docs and typed payload examples for market-aware universes |
| Compatibility path | `UniverseDefinition.from_legacy(...)` | Legacy strings: all/nyse/nasdaq/amex/sp500/custom/test | Ambiguous/non-extensible for regional markets | Keep adapter for transition, move callers to typed `universe_def` |
| Special-case watchlist | `backend/app/models/market_scan.py` (`ScanWatchlist.symbol=String(50)`) | Allows broad symbols (`TVC:DXY`, `BITSTAMP:BTCUSD`) | Separate domain with wider symbols already | Keep isolated; avoid regressing this flexibility |
| Tests/fixtures | Broad test suite under `backend/tests/**` and `frontend/src/**/*.test.*` heavily uses US symbols (`AAPL`, `NVDA`, etc.) | Limited non-US coverage | Migration regressions may go undetected | Add HK/JP/TW symbol fixtures in ST2/ST3 regression pack |

## Key Evidence Anchors

- `backend/app/models/stock.py`
- `backend/app/models/stock_universe.py`
- `backend/app/models/scan_result.py`
- `backend/app/models/provider_snapshot.py`
- `backend/app/models/theme.py`
- `backend/app/models/watchlist.py`
- `backend/app/services/theme_extraction_service.py`
- `backend/app/services/watchlist_import_service.py`
- `backend/app/api/v1/fundamentals.py`
- `backend/app/schemas/universe.py`
- `frontend/src/api/scans.js`

## Migration Checklist Hand-off to ST2

- [ ] Choose canonical symbol column width and apply consistently to all core symbol columns.
- [ ] Create Alembic migration expanding constrained columns/indexes.
- [ ] Update ORM models to match migrated widths.
- [ ] Replace US-only regex checks with market-aware resolution + active-universe validation.
- [ ] Remove hard `len(symbol) > 10` guardrails from API paths.
- [ ] Extend typed universe contracts for market-aware payloads.
- [ ] Add non-US fixtures/tests across API, service, and UI contract layers.
- [ ] Run upgrade/downgrade rehearsal and record timings/blockers.

## Notes for Migration PR Linking

This ST1 artifact is intended to be linked from the ST2 migration PR as the authoritative pre-change inventory.
