# Stock Scanner v1.1.0

Stock Scanner v1.1.0 expands the platform from four markets to five with full India support, harmonizes the per-market scan and refresh lifecycle, and adds richer per-row classification across sector, industry, themes, and group rank. Hero visuals and the screenshot guide have been refreshed to match the current UI.

## Highlights

### India market support

- Fifth supported market alongside US, Hong Kong, Japan, and Taiwan.
- 4,927-symbol universe loaded from official NSE/BSE feeds with 165 industry subgroups.
- Dedicated `data_fetch_in` and `user_scans_in` Celery queues, per-market cache warming schedules, and a `MarketTaxonomyService` loader that's wired end-to-end through the scan API and the React market selector.

### Per-market lifecycle harmonization

- Each market runs on its own exchange calendar (XNYS / XHKG / XTKS / XTAI / XNSE) with independent refresh queues and locks, so US and Asia hydrate in parallel without contention.
- Daily breadth and group-ranking pipelines now extend to HK, JP, TW, and IN — not just US.
- New gap-fill orchestrator backfills missing breadth and group-rank rows across all markets when a refresh cycle is interrupted.
- Bootstrap cache-only mode is hardened so first-run scans no longer fall back to live API calls when the cache is incomplete.

### Richer scan-results table

- New per-row classification columns: **GICS Sector**, **IBD Industry**, **Themes** (compact chip cluster with `+N` overflow), and **Group Rank** (bolded for top-20 groups).
- Company names now render under the ticker in the Symbol cell.
- Compact theme rendering keeps the wrap view from clipping when a stock carries multiple market themes.

### Performance

- Parallel cache-warm scans across markets cut the bootstrap warm-up window.
- Scanner data fetch parallelized across Finviz, yfinance, and the curl_cffi path with a shared circuit breaker.
- Optimized breadth and group-rank backfills reduce per-day refresh cost.

### Documentation and licensing

- Refreshed hero GIF demonstrates the cross-market workflow with the stock-detail popup (TradingView-style chart, scores, RS, growth, valuation, and Setup Engine pattern).
- README visuals re-captured against the current UI: market selector with India, breadth chart, scan results with new classification columns, market badges across US/HK/JP.
- Apache 2.0 license added.
- `docs/SCREENSHOT_GUIDE.md` updated with the new column list and per-asset capture specs.

## Deployment

Release images are published to GHCR under the `v1.1.0` tag:

- `ghcr.io/<owner>/stockscreenclaude-backend:v1.1.0`
- `ghcr.io/<owner>/stockscreenclaude-frontend:v1.1.0`

Deploy by setting `APP_IMAGE_TAG=v1.1.0` in `.env.docker`, then `docker-compose ... pull` and `up -d --no-build`. To roll back, set `APP_IMAGE_TAG=v1.0.0` and redeploy.

## Upgrade notes

- The bootstrap state machine now rejects `manual` scans against markets whose price cache is stale relative to the last completed trading day. If you were relying on automatic yfinance fallback for manual scans, expect HTTP 409 `market_data_stale` until the cache catches up.
- New per-market Celery queues (`user_scans_in`, `data_fetch_in`, `market_jobs_in`) are spawned automatically by `start_celery.sh` when `IN` is in `ENABLED_MARKETS` (default).
- `MARKET_TAXONOMY_SERVICE` paths now expect `data/india-deep.csv` to be present; the file ships with the repo.

Update this file before future semver tags so each GitHub release carries a maintained capability summary alongside the tagged image version.
