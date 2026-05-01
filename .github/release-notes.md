# Stock Scanner v1.1.2

Stock Scanner v1.1.2 expands the v1.1 multi-market release with Korea and mainland China support, improves first-run bootstrap reliability, and hardens market-data fallbacks for static and Docker deployments.

## Highlights

### Korea and China market support

- Adds Korea coverage for KOSPI and KOSDAQ, including KRX-backed universe listings, KRX price/fundamental fields, KRW currency formatting, market badges, and dedicated Celery queues.
- Adds mainland China A-share coverage for SSE, SZSE, and BJSE with AKShare/BaoStock-backed listings and OHLCV fallbacks.
- Extends market selectors, breadth views, group rankings, static market flags, benchmark registry, calendars, FX handling, and ticker validation across the seven-market set: US, HK, IN, JP, KR, TW, and CN.

### Bootstrap reliability and scan throughput

- Initial bootstrap now reports market-stage failures more accurately and isolates market queues so a failing non-primary market does not block unrelated market work.
- Bootstrap scans use a lighter default profile and cache-only safeguards to avoid expensive live-provider fallbacks while the cache is warming.
- Bulk scans can use symbol-level parallelism for non-cache-only runs, and Setup Engine detector execution is bounded to reduce slow per-symbol outliers.
- High-tight-flag detection was vectorized to reduce detector spikes on real 500+ bar histories.

### Market-data fallback hardening

- Korea official-universe refresh falls back to KRX listing finder data when daily ticker lists are unavailable, while preserving point-in-time correctness for historical dates.
- China listing fetches and OHLCV refreshes are bounded by timeout handling so slow AKShare requests can fail over instead of hanging a worker.
- China AKShare OHLCV failures are throttled after repeated transport errors to avoid hammering unstable upstream endpoints.
- pykrx runtime imports now include the setuptools dependency required by `pkg_resources`.

### Static site and release workflows

- Static-site and weekly-reference-data workflows now include Korea in the market matrix.
- Group-ranking cache misses now use benchmark cache fallbacks and report clearer missing-cache messages.
- Release notes and README deployment examples now point at `v1.1.2`.

## Deployment

Release images are published to GHCR under the `v1.1.2` tag:

- `ghcr.io/<owner>/stockscreenclaude-backend:v1.1.2`
- `ghcr.io/<owner>/stockscreenclaude-frontend:v1.1.2`

Deploy by setting `APP_IMAGE_TAG=v1.1.2` in `.env.docker`, then `docker-compose ... pull` and `up -d --no-build`.

## Upgrade notes

- Fresh bootstrap can take longer when several markets are enabled. Each selected market starts its own universe, price, fundamentals, breadth, group-rank, and scan work; smaller hosts should start with one primary market and enable more markets after the workspace is ready.
- Korea requires `pykrx` and the pinned setuptools runtime dependency included in this release.
- China data providers can throttle or disconnect under sustained load. The new timeout and backoff behavior should keep workers moving, but some CN symbols may still be skipped until the next refresh cycle.

Update this file before future semver tags so each GitHub release carries a maintained capability summary alongside the tagged image version.
