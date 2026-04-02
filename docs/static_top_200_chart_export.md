# Static Top-200 Chart Export Review

## Implemented Shape

The static site now exports a separate chart bundle for the top `200` symbols from the default daily scan ranking.

The existing chart stack already computes the moving averages and weekly aggregation client-side from OHLCV data:

- [get_price_history()](/Users/admin/StockScreenClaude/backend/app/api/v1/stocks.py#L267)
- [CandlestickChart.jsx](/Users/admin/StockScreenClaude/frontend/src/components/Charts/CandlestickChart.jsx#L300)

That means the static export only needs cached daily OHLCV bars, not pre-rendered chart images or precomputed EMA series.

## Proposed Shape

Export one manifest plus one JSON file per symbol:

- `static-data/charts/index.json`
- `static-data/charts/<SYMBOL>.json`

Current `index.json` shape:

```json
{
  "schema_version": "static-charts-v1",
  "generated_at": "2026-04-02T20:10:00Z",
  "as_of_date": "2026-04-02",
  "limit": 200,
  "symbols": [
    { "symbol": "NVDA", "rank": 1, "path": "charts/NVDA.json" }
  ]
}
```

Current per-symbol payload:

```json
{
  "schema_version": "static-charts-v1",
  "symbol": "NVDA",
  "rank": 1,
  "period": "6mo",
  "bars": [
    { "date": "2026-03-31", "open": 144.2, "high": 146.0, "low": 143.8, "close": 145.4, "volume": 28100200 }
  ],
  "stock_data": {
    "symbol": "NVDA",
    "company_name": "NVIDIA Corporation",
    "ibd_group_rank": 1,
    "ibd_industry_group": "Semiconductors"
  },
  "fundamentals": {
    "symbol": "NVDA",
    "description": "..."
  }
}
```

## Export Source

Use cached data only:

- read from `PriceCacheService.get_cached_only(symbol, period="2y")`
- bulk-export with `PriceCacheService.get_many_cached_only(symbols, period="2y")`
- trim to `6mo` during export
- read sidebar fundamentals from `FundamentalsCacheService.get_many_cached_only(symbols)`
- skip symbols with no cached price history instead of live-fetching during the static build

That keeps the static Pages run aligned with the current no-Redis, batch-first design.

## Expected Size

For `200` symbols with roughly `126` trading days of OHLCV:

- raw JSON per symbol: roughly `25 KB` to `60 KB`
- total raw size: roughly `5 MB` to `12 MB`
- gzip/compressed transfer size on Pages: roughly `1.5 MB` to `4 MB`

This is small relative to the scan dataset and well within GitHub Pages limits.

## UI Integration Cost

The static site now:

- reads `charts/index.json` plus per-symbol chart payloads
- reuses the existing chart transforms in [CandlestickChart.jsx](/Users/admin/StockScreenClaude/frontend/src/components/Charts/CandlestickChart.jsx#L300)
- reuses [StockMetricsSidebar.jsx](/Users/admin/StockScreenClaude/frontend/src/components/Scan/StockMetricsSidebar.jsx#L72) with exported scan-row and fundamentals metadata
- enables chart actions only for symbols present in the top-200 chart manifest

## Suggested Rollout

1. Export `charts/index.json` plus top-200 per-symbol `6mo` OHLCV files.
2. Include sidebar metadata (`stock_data` + `fundamentals`) in each per-symbol payload.
3. Reuse the current chart rendering code with a static data adapter.
4. Keep chart actions limited to symbols present in the exported manifest.
