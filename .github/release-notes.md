# Stock Scanner v1.6.0

Stock Scanner v1.6.0 introduces the Options Command Center: a focused view of options positioning, volatility, skew, and activity for liquid stocks already leading the equity scans. This release also expands market-breadth attribution, adds an RS heatmap to group rankings, and improves watchlist and static-site reliability.

## Highlights

### Options Command Center

- Selects the top 40 US Candidates and top 40 US Leaders independently, requires daily dollar volume above $100 million, and merges duplicate tickers while preserving both source ranks.
- Organizes the cohort into Gamma, Volatility, Skew, and Activity views, with sortable metrics and ticker-level strike charts, history, assumptions, and quality evidence.
- Calculates max pain, estimated net gamma exposure and gamma flip, call and put walls, ATM implied volatility, 25-delta skew, realized volatility, volatility risk premium, and contract-activity measures.
- Uses the earliest standard monthly Yahoo expiration 14–45 days away and publishes only when at least 90% of current symbols have core-valid chains.
- Preserves ticker history across temporary leadership gaps and reports unavailable or still-building metrics explicitly instead of substituting values.
- Supports opt-in live collection and a read-only static-site snapshot with history carried across deployments.

### Market breadth and contributor attribution

- Moves breadth calculations to market-calibrated, local-currency thresholds and provides a revision-3 rebuild, validation, cutover, and rollback path.
- Adds live and static contributor drilldowns so users can inspect the stocks and industry groups behind breadth readings.
- Persists contributor snapshots with calculation provenance and retains last-good static metadata across refreshes and deployments.
- Aligns static contributor inputs with the canonical breadth calculation and finalizes exports only after their source data is hydrated.

### Group rankings and watchlists

- Adds an RS heatmap to group rankings, with tones derived from the values displayed in each cell.
- Shows a stock's existing watchlist memberships in the detail view and keeps membership state synchronized after single or bulk changes.

## Deployment

Release images are published to GHCR under the `v1.6.0` tag:

- `ghcr.io/<owner>/stockscreenclaude-backend:v1.6.0`
- `ghcr.io/<owner>/stockscreenclaude-frontend:v1.6.0`

Set `APP_IMAGE_TAG=v1.6.0` in the deployment environment, pull the images, and recreate the application services using the normal Docker Compose deployment command.

## Upgrade notes

- Apply the included database migrations through revision `0034` using the normal deployment migration process. Revision `0033` adds breadth-contributor snapshots; revision `0034` adds Options Command Center persistence.
- Live options analytics are disabled by default. Set `OPTIONS_ANALYTICS_ENABLED=true` and recreate the API and workers to enable them.
- Options analytics are US-only and do not change the existing multi-market first-run bootstrap.
- The options job follows a successfully published daily US Feature snapshot on the existing `data_fetch_us` queue. No additional worker family is required, but the normal US data-fetch worker must be running.
- Historical options changes require five usable observations; IV percentile and rank require at least 20. The interface reports **Building history** until those thresholds are met.
- Yahoo options data is unofficial and best effort. It may be delayed, incomplete, or throttled and should be treated as research context rather than execution data.
- Existing breadth data is not silently converted to revision 3. Follow `docs/runbooks/market-breadth-revision-3-cutover.md` to rebuild, validate, activate, monitor, or roll back the new market-calibrated policy.

## What's Changed

- Use market-calibrated local thresholds for breadth by @xang1234 in https://github.com/xang1234/stock-screener/pull/349
- Add RS heatmap to group rankings by @xang1234 in https://github.com/xang1234/stock-screener/pull/350
- Add breadth contributor drilldowns by @xang1234 in https://github.com/xang1234/stock-screener/pull/351
- Fix static breadth contributor input parity by @xang1234 in https://github.com/xang1234/stock-screener/pull/352
- Export persisted breadth contributors by @xang1234 in https://github.com/xang1234/stock-screener/pull/354
- Fix stock-detail watchlist membership state by @xang1234 in https://github.com/xang1234/stock-screener/pull/355
- Fix static breadth contributor metadata retention by @xang1234 in https://github.com/xang1234/stock-screener/pull/356
- Add the Options Command Center by @xang1234 in https://github.com/xang1234/stock-screener/pull/358
- Document the Options Command Center by @xang1234 in https://github.com/xang1234/stock-screener/pull/360

**Full changelog:** https://github.com/xang1234/stock-screener/compare/v1.5.0...v1.6.0
