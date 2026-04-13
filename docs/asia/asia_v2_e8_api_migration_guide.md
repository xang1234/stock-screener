# ASIA v2 E8 API / Client Migration Guide

- Date: 2026-04-14
- Scope: `StockScreenClaude-asia.8` (E8 API and Frontend Multi-Market Productization)
- Audience: external API consumers + internal rollout reviewers (`StockScreenClaude-asia.11.*`)
- Covers: T1 (typed `universe_def`), T2 (frontend selector), T3 (market/USD fields), T4 (translation metadata), T5 (suffixed-ticker symbol format)
- Cross-references: [Legacy universe compat + deprecation policy](./asia_v2_legacy_universe_compat_deprecation_policy.md), [Flag matrix + rollback runbook](./asia_v2_flag_matrix_and_rollback_runbook.md)

## Goal

Give API clients a single, dated, copy-pasteable reference for every E8 contract change so they can plan adoption before the 2026-10-31 sunset and distinguish hard breaks from additive fields.

## Deprecation Timeline

| Milestone | Date |
|---|---|
| E8 contract changes announced | **2026-04-14** |
| Legacy `universe` request string sunset (shared with [legacy compat policy](./asia_v2_legacy_universe_compat_deprecation_policy.md)) | **2026-10-31** |
| `ScanListItem` flat universe fields sunset (**already removed** from responses as of T1 merge, 2026-04-13) | shipped |
| Earliest date the API may remove legacy `universe` request parsing | 2026-10-31 |

All E8 changes share a single sunset date (2026-10-31) to keep the client integration matrix simple. The only hard break on the response side — T1's flat `universe_*` field removal — has **already shipped**; the legacy _request_ field continues to work until sunset.

## Breakage Tiers

| Tier | What it means | Tasks |
|---|---|---|
| **Hard break (already shipped)** | Response fields removed; consumers must migrate before next deploy or UI breaks. | T1 `ScanListItem` response shape |
| **Soft break (validation tightened)** | Previously-tolerated malformed input now returns `422` instead of reaching provider/DB. | T5 stock detail + watchlist endpoints |
| **Additive** | New nullable response fields and new optional query params; old integrations keep working with `null`/defaults. | T3 scan result fields + USD filters, T4 translation metadata, T1 request-side `universe_def` |

---

## T1 — Typed `universe_def` Request/Response Contracts

**Status: shipped 2026-04-13 (`StockScreenClaude-asia.8.1`).**

### Request side (additive, legacy still accepted)

`POST /api/v1/scans` accepts either the legacy flat `universe: str` or the typed `universe_def: UniverseDefinition`. If both are present, `universe_def` wins (`schemas/scanning.py:32`).

**Before — legacy (still works until 2026-10-31):**

```json
POST /api/v1/scans
{
  "universe": "sp500",
  "screeners": ["minervini"]
}
```

**After — typed (recommended for new integrations):**

```json
POST /api/v1/scans
{
  "universe_def": {"type": "index", "index": "SP500"},
  "screeners": ["minervini"]
}
```

Legacy-to-typed mapping table: see the [deprecation policy](./asia_v2_legacy_universe_compat_deprecation_policy.md#legacy-to-typed-mapping).

### Response side (hard break — flat fields removed)

`GET /api/v1/scans`, `POST /api/v1/scans`, and `GET /api/v1/scans/{id}/status` return a nested `universe_def` object. Six flat fields were **removed** from `ScanListItem`: `universe`, `universe_type`, `universe_market`, `universe_exchange`, `universe_index`, `universe_symbols_count`.

**Before:**

```json
{
  "scan_id": "s_abc",
  "status": "completed",
  "universe": "sp500",
  "universe_type": "index",
  "universe_index": "SP500",
  "universe_symbols_count": 500,
  "total_stocks": 500,
  "passed_stocks": 42
}
```

**After:**

```json
{
  "scan_id": "s_abc",
  "status": "completed",
  "universe_def": {
    "type": "index",
    "index": "SP500"
  },
  "total_stocks": 500,
  "passed_stocks": 42
}
```

Client action: read `scan.universe_def.{type,market,exchange,index,symbols}` instead of the flat fields. `UniverseDefinition` schema lives at `backend/app/schemas/universe.py:44`.

### Legacy-path observability

When a client sends legacy `universe` on request, the API returns deprecation headers and increments Redis telemetry (`universe_compat:legacy_total`, per-value buckets, `universe_compat:legacy_last_seen_ts`). Operators can watch these trend to zero before sunset. See [deprecation policy §Legacy-Path Observability](./asia_v2_legacy_universe_compat_deprecation_policy.md#legacy-path-observability).

---

## T2 — Frontend Universe Selector (Market → Scope)

**Status: shipped 2026-04-13 (`StockScreenClaude-asia.8.2`).**

Primarily a UI change; relevant to API consumers only as the _canonical_ universe-payload shape the project emits.

- Frontend now emits **only** typed `universe_def` on scan creation (no more legacy `universe` strings from the bundled web app).
- Two-step picker: Market (US/HK/JP/TW/Test) → Scope (All / Exchange / Index).
- Universe counts in the UI are sourced from `GET /api/v1/universe/stats` → `by_market.{market}.counts.active` and `by_exchange`/`sp500` (not hard-coded), so third-party UIs should do the same to avoid drift.

Asia index membership (HSI, Nikkei 225, TWSE top indices) is intentionally deferred — tracked in `StockScreenClaude-7hwc`. Until then, HK/JP/TW clients should use `{"type": "market", "market": "HK"}` (and equivalents), not attempt `index`-type universes.

---

## T3 — Market Badges, Currency Context, and USD Filters

**Status: shipped 2026-04-13 (`StockScreenClaude-asia.8.3`). Backend plumbing only — frontend UI tracked in `StockScreenClaude-3axp`.**

### Additive response fields on `ScanResultItem`

Five new nullable fields on every row (`schemas/scanning.py:189-197`):

| Field | Type | Source | Meaning |
|---|---|---|---|
| `market` | `"US" \| "HK" \| "JP" \| "TW" \| null` | `stock_universe` | Market identity |
| `exchange` | `str \| null` | `stock_universe` | Native exchange code |
| `currency` | `str \| null` | `stock_universe` | Local currency (HKD, JPY, TWD, USD) |
| `market_cap_usd` | `float \| null` | `stock_fundamentals` | FX-normalized market cap |
| `adv_usd` | `float \| null` | `stock_fundamentals` | FX-normalized average dollar volume |

**Critical semantic note:** the existing `market_cap` and `volume` fields remain in **local currency**. Cross-market comparisons must use `*_usd`.

### New query params on `GET /api/v1/scans/{id}/results`

From `backend/app/api/v1/scan_filter_params.py:146-151`:

- `min_market_cap_usd`, `max_market_cap_usd` — integer USD
- `min_adv_usd`, `max_adv_usd` — integer USD
- `markets=US,HK` — comma-separated market codes

**Example — HK + JP stocks, $1B+ USD cap:**

```
GET /api/v1/scans/s_abc/results?markets=HK,JP&min_market_cap_usd=1000000000
```

### CSV export columns

Export (`GET /api/v1/scans/{id}/export?format=csv`) now includes `Market`, `Exchange`, `Currency`, `Market Cap (USD)`, `ADV (USD)`. The pre-existing `Market Cap` header was **renamed to `Market Cap (local)`** to make the currency explicit — downstream spreadsheet jobs reading by header name must update.

### Backward compatibility

All fields are `Optional[...] = None`, so US-only integrations that ignore the new fields keep working unchanged. The `market` filter defaults to no restriction.

---

## T4 — Theme Translation Metadata on Content + Mentions

**Status: shipped 2026-04-13 (`StockScreenClaude-asia.8.4`).**

Additive nullable fields surfaced on two response schemas so clients can render original + translated text with language markers.

### `ThemeMentionDetailResponse` — `backend/app/schemas/theme.py:377`

```json
{
  "mention_id": 12345,
  "theme_id": "t_xyz",
  "excerpt": "ソニーは好調な四半期…",
  "source_language": "ja",
  "translated_excerpt": "Sony posted strong quarterly…",
  "translated_raw_theme": "AI chip demand",
  "translation_metadata": {
    "provider": "minimax",
    "model": "...",
    "source_language": "ja",
    "target_language": "en",
    "confidence": 0.92,
    "translated_at": "2026-04-12T10:30:00Z"
  }
}
```

### `ContentItemWithThemesResponse` — `backend/app/schemas/theme.py:949`

```json
{
  "id": 9876,
  "title": "ソニー 決算発表",
  "content": "...",
  "source_language": "ja",
  "translated_title": "Sony Earnings Announcement",
  "translated_content": "...",
  "translation_metadata": { "...": "see TranslationMetadata" }
}
```

### Rules clients must respect

- **English sources return `source_language=null` and no translated fields.** Do not render a language chip in that case.
- `translation_metadata` uses `model_config = ConfigDict(extra="allow")` — future T7.x metadata additions won't bump the schema version. Do not validate with `extra="forbid"`.
- Mention-level `translation_metadata` falls back to the parent content-item's metadata when unset. Clients that only need a provenance stamp can read either.
- CSV export on `GET /api/v1/content-items` now includes `Source Language`, `Translated Title`, `Translated Content` columns.

The normative pipeline contract behind these fields is [ADR ASIA-E4: TranslationPipeline](./adr_asia_e4_translation_pipeline_v1.md).

---

## T5 — Symbol-Format Compatibility for Suffixed Non-US Tickers

**Status: shipped 2026-04-14 (`StockScreenClaude-asia.8.5`).**

The canonical symbol contract now accepts `.HK`, `.T`, `.TW`, `.TWO` suffixed tickers uniformly across all user-facing endpoints.

### Canonical shape

- Regex: `^[A-Z0-9][A-Z0-9.\-]{0,19}$` (20-char max, matches `stock_universe.symbol` `VARCHAR(20)`)
- Supported suffixes: `.HK` (Hong Kong), `.T` (Tokyo), `.TW`, `.TWO` (Taiwan)
- Module: `backend/app/services/symbol_format.py`

Normalizer applies `strip() → lstrip("$") → upper()` before matching. Lowercase input and `$`-cashtag prefixes from social feeds are accepted.

### Endpoint behavior changes (soft break)

All `/api/v1/stocks/{symbol}/...` routes (`/info`, `/fundamentals`, `/technicals`, `/industry`, `/chart-data`, `/decision-dashboard`, `/peers`, `/history`, `/validation`) and `GET /api/v1/stocks/{symbol}` now validate shape **before** DB/provider work:

| Input | Before (pre-T5) | After (T5) |
|---|---|---|
| `NVDA` | 200 / 404 | 200 / 404 (unchanged) |
| `0700.HK` | **varies** (some endpoints 500'd on upstream) | 200 / 404 (consistent) |
| `NV DA` (malformed) | 500 (reached data provider) | **422** — `{"detail":"Invalid symbol format: 'NV DA'"}` |
| `X * 21` chars | 500 / 404 | **422** |
| `日経` (non-ASCII) | 500 | **422** |

Watchlist endpoints (`POST /api/v1/user-watchlists/{id}/items`):

| Input | Response |
|---|---|
| `{"symbol": "0700.HK"}` (in active universe) | `200` |
| `{"symbol": "aapl"}` | `200` — normalized to `"AAPL"` |
| `{"symbol": "NV DA"}` | `422` — invalid format |
| `{"symbol": "UNKNOWN.HK"}` (format-valid, not in universe) | `400` — "not in the active stock universe" |

**Client action:** if your code previously relied on `5xx` to detect malformed input, switch to `422`. The discriminator between `422` (malformed) and `404`/`400` (well-formed but unknown) is now clean and stable.

### Bulk add silently drops invalid entries

`POST /api/v1/user-watchlists/{id}/items/bulk` returns only the symbols that were both format-valid and present in the active universe. Invalid/unknown entries are dropped without error — diff the request and response arrays client-side if you need a rejected-set.

### Out of scope

Theme-extraction LLM ticker validation (`multi_market_ticker_validator`) keeps its stricter 12-character regex on purpose — narrower input window limits hallucination blast radius. No client action needed.

---

## Client Adoption Checklist

For any client integrating against scans, watchlists, or theme content:

- [ ] **T1 response:** replace reads of `scan.universe` / `scan.universe_type` / `scan.universe_market` / `scan.universe_exchange` / `scan.universe_index` / `scan.universe_symbols_count` with `scan.universe_def.*`. (Already breaking as of 2026-04-13.)
- [ ] **T1 request (by 2026-10-31):** emit `universe_def` instead of `universe` string. Verify zero traffic on `universe_compat:legacy_total` Redis counter for your client.
- [ ] **T3:** treat `market_cap` / `volume` as local-currency; use `market_cap_usd` / `adv_usd` for cross-market comparison. Update any CSV header parsers that matched `"Market Cap"` exactly (now `"Market Cap (local)"`).
- [ ] **T4:** when `source_language != null` and `source_language != "en"`, render the translated field with a language marker. Parse `translation_metadata` with extra-keys-tolerant validation.
- [ ] **T5:** accept `.HK`/`.T`/`.TW`/`.TWO` suffixes in client-side symbol inputs. Handle `422` as malformed-input vs. `400`/`404` as not-found. Remove any client-side regex narrower than `^[A-Z0-9][A-Z0-9.\-]{0,19}$`.

## Rollout Cross-References

- Feature-flag gating for per-market UI/API exposure: see [flag matrix + rollback runbook §Flag Taxonomy](./asia_v2_flag_matrix_and_rollback_runbook.md#flag-taxonomy).
- Dress-rehearsal gates that must pass before re-enabling: `StockScreenClaude-asia.11.*` (references this guide for client-migration evidence).
- Unsupported-field transparency surfaces (client-facing reason codes when data absent): tracked in `StockScreenClaude-asia.8.7`.

## Changelog

| Date | Change | Source |
|---|---|---|
| 2026-04-13 | T1 shipped: `universe_def` first, flat `ScanListItem` fields removed | `StockScreenClaude-asia.8.1` |
| 2026-04-13 | T2 shipped: two-step frontend selector, typed-only emission | `StockScreenClaude-asia.8.2` |
| 2026-04-13 | T3 shipped: `market`/`exchange`/`currency`/`*_usd` on `ScanResultItem`; USD filter params; CSV header rename | `StockScreenClaude-asia.8.3` |
| 2026-04-13 | T4 shipped: translation metadata on content + mention responses | `StockScreenClaude-asia.8.4` |
| 2026-04-14 | T5 shipped: uniform symbol-format validation across stock + watchlist endpoints | `StockScreenClaude-asia.8.5` |
| 2026-04-14 | This guide published | `StockScreenClaude-asia.8.6` |
