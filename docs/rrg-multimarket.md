# RRG Multi-Market Expansion — Design

**Status:** Proposed · **Scope:** Extend the Relative Rotation Graph (RRG) beyond US to the other group-rankings markets.

The RRG (`backend/app/services/rrg_service.py`, `frontend/src/components/Charts/RRGChart.jsx`) currently renders for **US only**. This document records *why* (it's a data-source coupling, not a classification gap for most markets), the per-market classification reality, and a phased plan to expand it cleanly.

## Background

RRG plots each industry **group** (or a **sector** roll-up) by RS-Ratio (x) vs RS-Momentum (y) with a weekly tail. It is rendered by the shared `RRGChart` on both the live Group Rankings page and the static site; the static exporter (`static_site_export_service._build_groups_rrg_payload`) bakes a per-market `groups_rrg.json` and gates `features.rrg` on availability, so **no frontend or static-export change is needed** to light up a new market — the moment `RRGService.get_rrg_scopes(market=X)` returns non-empty, the bundle and toggle appear automatically.

## The three prerequisites

A market needs all three to render RRG. Today only US has all three.

| # | Prerequisite | Current source | US | Non-US |
|---|---|---|---|---|
| 1 | **Group classification** (symbol → industry_group) | IBD CSV / deep CSVs / hybrid classifier | ✅ | varies (see matrix) |
| 2 | **Daily RS history** (~30 weekly `avg_rs_rating` points for full tails) | `IBDGroupRank` table | ✅ | ❌ table is US-populated only |
| 3 | **Sector data** (for the Sectors toggle) | `get_group_sector_map` → `IBDIndustryGroup ⋈ StockUniverse.sector` | ✅ (Finviz GICS) | ❌ wrong source for curated markets (below) |

## The decisive architectural finding: group membership lives in two disjoint stores

Non-US curated taxonomies are **never persisted** to `ibd_industry_groups`. They are parsed from committed deep CSVs into `MarketTaxonomyService` (in-memory) and served at runtime. The read paths already branch on this (`ibd_industry_service.py:241-300`):

| Store | Markets | Read by |
|---|---|---|
| `ibd_industry_groups` **table** | US; CA/DE/SG/MY (only if the hybrid classifier ran — sole writer is `ibd_classification_bundle.py:139`) | US group rankings; **`RRGService.get_group_sector_map`** |
| `MarketTaxonomyService` (in-memory CSV) | HK, IN, JP, TW, KR, CN | non-US group rankings (each scan row already carries `ibd_industry_group`) |

`_market_has_curated_taxonomy(market)` is simply `bool(groups_for_market(market))` — true for any market the taxonomy service loaded.

**Consequence:** `get_group_sector_map` joins the *table*, so it returns **empty for HK/IN/JP/TW/KR/CN** — precisely the markets with the best group classification. The RRG **Sectors** toggle is therefore broken for them by *source*, independent of data depth. (CA/DE/SG/MY use the table and would join fine if the classifier ran, but their `StockUniverse.sector` is empty → degraded.)

### Native sectors already exist (mostly) but aren't exposed

`MarketTaxonomyEntry` carries a `sector` field, captured by the loaders for several markets — but there is **no `sector_for_group` accessor**, and two loaders drop it:

| Market | Native sector captured? | Source column |
|---|---|---|
| IN | ✅ | `Industry (Sector)` |
| JP | ✅ | `TSE 17-Sector` |
| KR / CN | ✅ (but stub data — 30 / 7 rows) | `Sector` |
| CA | ✅ (CSV not yet shipped) | `Sector` |
| **HK** | ❌ **dropped** by `_load_hk` (`sector=None`) | `HSICS Sector` (present in `hk-deep.csv`) |
| **TW** | ❌ none | CSV has no sector column |
| US | ❌ (uses Finviz GICS roll-up) | — |

## Per-market readiness matrix

Static workflow builds 12 markets (`.github/workflows/static-site.yml` matrix); `market_codes_with_capability("group_rankings")` declares 8 (US, HK, IN, JP, KR, TW, CN, CA).

| Market | Group classification | Native sector | RS history (#2) | RRG readiness |
|---|---|---|---|---|
| **US** | ✅ IBD CSV (10.1k symbols) | GICS via Finviz | ✅ `IBDGroupRank` | **Ready** |
| **HK** | ✅ `hk-deep.csv` (9.9k) | ✅ in CSV but dropped by loader | needs provider | **Tier A** |
| **JP** | ✅ `kabutan_themes_en.csv` (1.8k) | ✅ TSE 17-Sector | needs provider | **Tier A** |
| **IN** | ✅ `india-deep.csv` (4.9k) | ✅ Industry (Sector) | needs provider | **Tier A** |
| **TW** | ✅ `taiwan-deep.csv` (1.1k) | ❌ none | needs provider | **Tier A−** (sector TBD) |
| **KR** | ⚠️ stub (`korea-deep.csv`, 30 rows) | ✅ col (no data) | needs provider | **Tier B (data)** |
| **CN** | ⚠️ stub (`china-deep.csv`, 7 rows) | ✅ col (no data) | needs provider | **Tier B (data)** |
| **CA** | ⚠️ classifier-only (CSV not shipped) | TMX baseline | needs provider | **Tier C** |
| **DE / SG / MY / AU** | ❌ none curated (AU not declared) | ❌ | needs provider | **Tier C** |

## Design — two independent workstreams

These are decoupled; either can ship alone (Groups scope works without Sectors).

### A. Groups scope — per-market RS-history provider

Mirror how the live Group Rankings page already serves non-US: `MarketGroupRankingService` reads per-market `avg_rs_rating` history from `FeatureRun` snapshots. Make `RRGService._fetch_inputs` pick a provider by market instead of always querying `IBDGroupRank`:

- **US:** keep the existing batched `IBDGroupRank` query.
- **Non-US:** add `MarketGroupRankingService.get_all_groups_history(market, days)` — compute every group's `avg_rs_rating` from each loaded `FeatureRun` in **one pass** (avoid the per-group N+1 that `get_group_history` would incur), yielding the same `(industry_group, date, avg_rs_rating, num_stocks)` tuples `compute_group_rrg` already consumes.

History depth then equals FeatureRun publication depth (~180d if published daily → ~36 weekly points, enough for full tails). The pure math (`compute_group_rrg`) and the static/ frontend plumbing are unchanged.

> Alternative considered: backfill `IBDGroupRank` per market via the daily group-rank task. Rejected as the primary path — it duplicates history the FeatureRun store already holds and adds per-market scheduling; keep `IBDGroupRank` as the US source only.

### B. Sectors scope — taxonomy-sourced sector map

Route `get_group_sector_map` through the store that actually owns each market's groups, exactly like `get_group_symbols`/`get_all_groups`:

1. Add `MarketTaxonomyService.sector_for_group(market, group)` — majority-vote of member entries' `entry.sector` (a **native** sector, strictly better than a GICS roll-up).
2. Branch `RRGService.get_group_sector_map`: curated market → native sector from the taxonomy service; US / classifier markets → keep the current `IBDIndustryGroup ⋈ StockUniverse.sector` path.

## Per-market data fixes

- **HK:** one-line loader fix — `_load_hk` should capture `HSICS Sector` instead of `None`.
- **TW:** no sector column → add one to `taiwan-deep.csv`, or leave the Sectors toggle hidden (the chart already degrades gracefully; Groups scope still works).
- **KR / CN:** the deep CSVs are stubs (30 / 7 rows). Source a real taxonomy (KRX sector/industry; CSI / SW industry for CN A-shares) before enabling — the existing group rankings for these markets are equally thin.
- **CA / DE / SG / MY / AU:** depend on the hybrid classifier (`IBDClassificationService`) populating `ibd_industry_groups` + a GICS sector backfill (`scripts/backfill_universe_sector_industry.py`). Lowest confidence; ship a curated CSV if these become a priority.

## What needs no change

- **Static export** (`_export_market_bundle`): already writes `groups_rrg.json` + `assets.groups_rrg` and sets `features.rrg` only when `get_rrg_scopes` returns non-empty.
- **Frontend** (`RRGChart`, `RRGViewToggle`, `useRRGFilters`): scope/market-agnostic; the toggle is gated on the manifest asset.

## Rollout order

1. **Workstream A** (history provider) — unblocks the Groups scope for all curated markets at once.
2. **HK → JP → IN** (Workstream B + HK loader one-liner) — strong groups *and* native sectors; minimal work.
3. **TW** — Groups now; Sectors once a sector source exists.
4. **KR / CN** — fix the stub taxonomies first, then enable.
5. **CA / others** — only if prioritized; classifier + sector backfill.

## Risks / open questions

- **CI history depth:** the static export seeds a fresh DB from release artifacts. Confirm the per-market `daily-price-data` / `weekly-reference-data` releases carry ≥ ~6 months, or non-US tails will be mostly `is_provisional`.
- **Stub taxonomies (KR/CN):** declaring `group_rankings=true` with 30 / 7 classified symbols produces misleading rankings *and* RRGs; treat as a data-quality blocker, not just an RRG one.
- **FeatureRun publication cadence per market:** tails assume ~daily publication; sparse cadence shortens usable history.

## Verification (per market enabled)

- Backend: `GET /v1/groups/rrg?market=<M>&scope=groups` returns groups with multi-month tails (few `is_provisional`); `scope=sectors` returns the market's native sectors.
- Static: a real export writes `static-data/markets/<m>/groups_rrg.json` and the manifest entry has `features.rrg=true`.
- Frontend: the Groups tab shows the RRG toggle for `<M>`; Sectors toggle present only where a sector source exists.
