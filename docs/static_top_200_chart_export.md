# Static Top-200 Chart Export Review

## Recommendation

Add a separate static chart bundle for the top `200` symbols from the default daily scan ranking.

The existing chart stack already computes the moving averages and weekly aggregation client-side from OHLCV data:

- [get_price_history()](/Users/admin/StockScreenClaude/backend/app/api/v1/stocks.py#L267)
- [CandlestickChart.jsx](/Users/admin/StockScreenClaude/frontend/src/components/Charts/CandlestickChart.jsx#L300)

That means the static export only needs cached daily OHLCV bars, not pre-rendered chart images or precomputed EMA series.

## Proposed Shape

Export one manifest plus one JSON file per symbol:

- `static-data/charts/index.json`
- `static-data/charts/<SYMBOL>.json`

Suggested `index.json` shape:

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

Suggested per-symbol payload:

```json
{
  "schema_version": "static-charts-v1",
  "symbol": "NVDA",
  "period": "6mo",
  "bars": [
    { "date": "2026-03-31", "open": 144.2, "high": 146.0, "low": 143.8, "close": 145.4, "volume": 28100200 }
  ]
}
```

## Export Source

Use cached price history only:

- read from `PriceCacheService.get_cached_only(symbol, period="2y")`
- trim to `6mo` during export
- skip symbols with no cached data instead of live-fetching during the static build

That keeps the static Pages run aligned with the current no-Redis, batch-first design.

## Expected Size

For `200` symbols with roughly `126` trading days of OHLCV:

- raw JSON per symbol: roughly `25 KB` to `60 KB`
- total raw size: roughly `5 MB` to `12 MB`
- gzip/compressed transfer size on Pages: roughly `1.5 MB` to `4 MB`

This is small relative to the scan dataset and well within GitHub Pages limits.

## UI Integration Cost

Low-to-moderate for charts only:

- add a static chart client that reads `charts/index.json` and `charts/<SYMBOL>.json`
- reuse the existing chart transforms in [CandlestickChart.jsx](/Users/admin/StockScreenClaude/frontend/src/components/Charts/CandlestickChart.jsx#L300)
- limit the chart icon/modal to symbols present in the top-200 chart manifest

Moderate-to-high for the full current modal:

The live modal also pulls sidebar data beyond OHLCV:

- chart metadata from [get_chart_data()](/Users/admin/StockScreenClaude/backend/app/api/v1/stocks.py#L194)
- fundamentals from `getStockFundamentals`
- group detail from `getGroupDetail`

If the static site should match the live modal, the export would also need:

- one static per-symbol metadata payload for the same fields returned by `chart-data`
- optional static group detail payloads for the displayed group sidebar

## Suggested Rollout

1. Export `charts/index.json` plus top-200 per-symbol `6mo` OHLCV files.
2. Enable chart icons only for symbols present in that manifest.
3. Reuse the current chart rendering code with a static data adapter.
4. Add static sidebar metadata only if the chart-only modal proves useful.
