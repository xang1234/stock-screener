# Static Site Guide

The static site is a read-only GitHub Pages demo built from pre-baked JSON bundles. It is useful for sharing a daily snapshot without running the live FastAPI/PostgreSQL/Redis/Celery stack.

Static demo: [https://xang1234.github.io/stock-screener/](https://xang1234.github.io/stock-screener/)

## What Works

- Daily Snapshot from exported market data.
- Static Scan view with read-only results and chart drill-ins.
- Breadth view from exported breadth JSON.
- Group Rankings and Relative Rotation Graph from exported group/sector data.
- Static market selection backed by the exported markets.

## What Is Read-Only Or Missing

Static mode does not call `/api` and does not run background jobs. It does not support:

- server login,
- first-run bootstrap,
- live scans or cache refresh,
- watchlist writes,
- theme pipeline runs or review actions,
- Assistant/chatbot workflows,
- Operations console,
- scheduled task controls,
- mutable filter presets or user settings.

Use the live app when you need interactive backend workflows.

## How It Differs From The Live App

| Area | Static Site | Live App |
|------|-------------|----------|
| Data | Pre-exported daily JSON bundles | PostgreSQL-backed runtime with live refresh workers |
| Auth | None | Optional/required server password session |
| Scan | Read-only exported results | User-triggered multi-screener scans |
| Watchlists | Snapshot display only | CRUD, folders, ordering, chart actions |
| Themes | Snapshot display only when exported | Pipeline runs, review queues, alerts, source controls |
| Assistant | Unavailable | Feature-gated chat and research tools |
| Operations | Unavailable | Runtime activity, telemetry, queue/job controls |

## Relative Strength Data Contract

Static exports use the `static-site-v3` schema and consume the same stored canonical Market RS snapshots as the live app. A Scan row carries overall, 1M, 3M, and 12M RS from `balanced-horizon-percentile-v2`; it does not recalculate RS from the exported subset. The Market manifest records the RS formula version, as-of date, eligible-universe size, and Market RS run ID.

Scheduled and manual Static Site workflows calculate balanced Market RS independently for every Market in their fresh private build databases before Feature, Group, and RRG generation.

Balanced activation validates every declared Scan chunk and requires exact formula, Market RS run, RS date, and universe-size metadata on the Scan manifest, chunks, and rows. Its approval fingerprint covers the root manifest and complete `markets/<market>` tree, rather than only `manifest.json`.

`markets/<market>/groups.json` is an exact export of the stored Group snapshot for that formula/run. It includes Group overall, 1M, and 3M RS, with Group Rank based only on overall RS. The static Group table exposes separate **1M RS** and **3M RS** columns, matching the live table.

RRG rolling-history assets use `static-rrg-history-v4` and identify their RS formula version. The weekly RS-Ratio/RS-Momentum transformation is unchanged, but history from different formulas is never merged. A fallback market artifact or RRG-history bundle is rejected when its schema does not match or it does not use `balanced-horizon-percentile-v2`. The fallback downloader (`backend/app/scripts/download_static_market_fallbacks.py`) skips such an artifact and keeps the market's newest balanced one. Until enough balanced history exists, RRG reports insufficient history.

## Breadth Data Contract

Breadth-capable static market artifacts must contain a revision-3 `breadth.json` payload and a `breadth-r3` source marker. The combiner rejects revision-1 or revision-2 breadth artifacts even when their Market RS formula is otherwise compatible. If no current revision-3 artifact exists, that market's breadth is unavailable; the publisher does not silently reuse an older breadth formula.

Revision 3 uses the same fixed local-market StockBee thresholds as the live app and has no historical-FX input. Markets whose catalog capability disables breadth (currently AU, SG, and MY) publish their other supported static sections without requiring breadth or breadth-derived exposure.

Contributor drilldowns are optional additive assets. When available, a market entry advertises `assets.breadth_contributors.index_path`; the index uses `breadth-contributors-v1` and references at most 20 newest-first per-session documents under `markets/<market>/breadth/contributors/`. Each stock is stored once with all qualifying signals plus its frozen company name and IBD group. The combiner validates every document and reconciles its 11 supported signal counts to `breadth.json`. Missing legacy assets remain valid, while an invalid contributor asset is omitted without removing the market or its aggregate breadth page.

Static builds preserve contributor display metadata in the dedicated `breadth-contributor-metadata-data` release. Each breadth-enabled market owns a canonical `breadth-contributor-metadata-<market>.json.gz` asset containing only company names and IBD groups for its newest 20 contributor sessions, plus a `.previous` recovery asset that protects interrupted replacements. On the first run, all retained sessions are bootstrapped from the current enriched universe and classification data. Later runs restore that state, keep existing date/symbol metadata frozen, and bootstrap only newly encountered sessions or symbols. Restored bytes are validated for gzip, schema, and market identity. The new state must pass non-empty name and group coverage checks and publish successfully before the corresponding market artifact can replace its last-good fallback. A missing canonical asset uses the validated recovery asset when present; only the absence of both is a safe first publication. An uncertain restore or failed state upload suppresses the current market artifact.

The static frontend loads the small index with the market bundle and fetches a date document only after a user clicks an advertised nonzero Recent History count. The live and static dialogs therefore use the same schema and do not recalculate breadth formulas in the browser.

This rolling release is a static-build continuity mechanism only. The live app continues to read contributor metadata from its own PostgreSQL snapshots and does not depend on GitHub Releases.

## Forward-Return Scoreboard

A market entry can advertise `assets.scoreboard = {"path": "markets/<m>/scoreboard.json"}`, a small file of what happened after a stock newly entered each preset's published matches. It adds no schema bump. A market never depends on it: the combiner validates it apart from required assets, so an unparseable or wrong-shape file loses only its `assets.scoreboard` entry, with a warning.

Past matches cannot be rebuilt, because fundamentals keep no history. So the `scoreboard-ledger-data` release stores what the site published, one line per first-published date, in `scoreboard-ledger-<m>-v1.jsonl.gz` with `scoreboard-ledger-<m>-v1.previous.jsonl.gz` as the prior copy. The ledger stores symbols and a hash of each preset's `filter_expression`. It never stores returns or entry prices, because every build recomputes them from current prices.

Three workflows touch the ledger:

- The `record-ledger` job in `static-site-publish.yml` runs after `deploy`, with the same `if`, so it records what actually deployed. Its `published_at` is the time the deploy job finished, so a late or rerun record job cannot move a pick's entry. It holds `contents: write`, installs nothing and runs `backend/app/scripts/scoreboard_ledger.py`, which is stdlib only. A date at or before the ledger's end uploads nothing. It goes red for a market whose extraction failed, whose day cannot have been published (dated past the publish UTC date plus one day, or holding zero presets), or whose canonical lost a line `.previous` holds.
- `scoreboard-ledger-recover.yml` inserts one lost day from a retained artifact. It is dispatched by hand and shares the production publisher's concurrency group, so it cannot overlap `record-ledger`.
- The `Build market scoreboard` step in `static-site.yml` downloads the ledger and runs `backend/app/scripts/build_static_scoreboard.py` as its own process, with `continue-on-error` and a 20 minute cap. This workflow never writes the release. If the step fails, `manifest.market.json` stays unchanged and the market publishes without the asset.

Returns follow these rules:

- **A pick.** A pick is a symbol newly entering a preset: in that day's matches and not in the preset's matches on its previous ledger day. A baseline day yields none. A preset's baselines are its first ledger day, a day whose `definition` differs from its previous ledger day's, and a day whose origin crosses from `artifact_pre_stage_fix` to anything else, because those changes move stocks onto the list without any stock newly qualifying. `artifact` to `live` is not a baseline, because that day's entrants are real.
- **Window.** Only ledger days within 365 calendar days of `as_of_date` are scored, with the day before them as the baseline. `scored_since` is the first scored day and `recording_since` the ledger's first. Days stay in the ledger.
- **Entry and exit.** Entry is the open of the first session that opens after publication. Exit is the close of the Nth session, for N of 5, 20 and 60. Returns are in local currency, price only, with no costs.
- **Prices.** The builder reads `stock_prices` directly, capped at the export's `as_of_date`, so it cannot see a partial bar from a session in progress and cannot reach a provider. A recent IPO still scores, because there is no 50-bar minimum.
- **Overlap.** A symbol that re-enters inside its counted window is suppressed. The rule runs per preset across the scored days, so a definition or era change does not reset it, and a symbol in two presets counts once in each.
- **Cohorts.** Numbers are never pooled across `(preset, definition, origin)`. `origin` is `live`, `artifact` (read from a retained build artifact, whose `created_at` only approximates publication) or `artifact_pre_stage_fix` (the same, for an artifact built before the stage fix merged at 2026-10-21T23:05:12Z, #671). The era follows the build time, so US as-of 2026-10-21, built after the merge, is `artifact` while Asian markets' as-of 2026-10-21, built before it, is `artifact_pre_stage_fix`.
- **Delisted stocks.** A symbol with no price counts in `skipped_missing_history`. Stocks that later delist drop out this way, which can flatter results, so the count is published beside the returns.

Each horizon block has `sample_size`, `signal_dates`, `effective_dates`, `pending_windows`, `oldest_open_sessions`, `thin`, `positive_rate` and `beat_benchmark_rate` (fractions from 0 to 1), `median_return_pct`, `median_excess_return_pct`, `excess_sample_size`, `skipped_missing_history` and `suppressed_overlapping`. There are no averages, because one bad bar can move an average by many points. `beat_benchmark_rate` is the share of benchmark-covered closed events with excess above zero. `effective_dates` is `(sum n)^2 / sum(n^2)` over the closed events' entry sessions, rounded to one decimal, so one busy day counts as about one date. Days whose picks enter on the same session, such as a day published late and the day after it, count as one. `thin` means fewer than 20 closed events or an `effective_dates` below 5. `oldest_open_sessions` is null when no window is open. Statistics with no closed window are null.

`missing_sessions` counts completed sessions between `recording_since` and `ledger_through` with no ledger line. To seed the release, recover a lost day or repair a corrupt asset, see the [scoreboard ledger runbook](runbooks/scoreboard-ledger.md).

## Market History Archive

The site publishes only the newest session, and past fundamentals, ratings and screen matches cannot be rebuilt later. So each market has a Release, `market-history-<m>-v1`, that holds one immutable file per published session. The Release is a prerelease created with `--latest=false`.

Each session file is `<m>-<YYYY-MM-DD>.json.gz`, gzip with `mtime=0`:

```json
{
  "as_of_date": "2026-11-06",
  "columns": ["rs_rating", "symbol"],
  "market": "US",
  "origin": "live",
  "presets": {
    "minervini": {
      "definition": "241a0f65f493c25e",
      "filter_expression": {},
      "name": "Minervini",
      "symbols": ["AAA"]
    }
  },
  "published_at": "2026-11-07T02:03:30Z",
  "rows": [[83.0, "AAA"]],
  "schema_version": "market-history-v1"
}
```

- `columns` is sorted and always holds `symbol`. A row that lacks a field holds `null` there.
- `rows` is sorted by symbol. Rows keep every published field except `price_sparkline_data`, `rs_sparkline_data` and `field_availability`.
- `origin` takes the scoreboard ledger's values: `live`, `artifact` or `artifact_pre_stage_fix`.
- `definition` is the ledger's hash of `filter_expression`, so a reader can see a rule change.

`index.json` lists the sessions, for example `{"schema_version":"market-history-index-v1","market":"US","sessions":["2026-11-05","2026-11-06"]}`. It is derived from the Release listing and never authoritative. Every run compares it with the listing and rewrites it when they differ, so a crash between the two uploads heals on the next run. `gh release upload --clobber` deletes before it uploads, so a reader can briefly find no index.

These rules decide what the archive holds:

- **The first publication of a session wins.** No job overwrites a session asset, so `published_at` and the content mean what the site first showed. A later correction to the same session does not appear.
- **Any absent date can be added**, including one older than the newest, because each session is its own file.
- **Failures are per market.** One market going red does not stop the others, and the archive never affects the deploy or the ledger.
- **Mid-session behavior.** The archive reads the built site only. It fetches no prices and calls no provider, so a session means whatever the site published for that `as_of_date`.

Two workflows write the Releases:

- The `build` job in `static-site-publish.yml` extracts the session from the built site and uploads it as the `market-history-days` artifact for 7 days. Both steps are `continue-on-error`. The `record-history` job then runs after `deploy`, with the same `if`, using the deploy's finish time as `published_at`. It holds `contents: write`, installs nothing and runs `backend/app/scripts/market_history.py`, which is stdlib only.
- `market-history-backfill.yml` adds absent sessions from retained artifacts. It is dispatched by hand, one market at a time, from the default branch. It has its own concurrency group, because a shared group would let a pending backfill replace a pending publish.

To run a backfill, delete a bad session or clear an incomplete asset, see the [market history runbook](runbooks/market-history.md).

## Maintaining Static Assets

- Keep the static page tour GIF aligned with the exported static routes.
- Static builds (`VITE_STATIC_SITE=true`) get search and link-preview `<meta>` tags from `staticSiteMetaPlugin` in `frontend/vite.config.js`, using the title and description in `frontend/src/static/siteMeta.js`; the live app's `index.html` is unchanged. `og:image` points at `frontend/public/og-image.png` (1200×630) under `VITE_SITE_URL`, which the publish workflow sets. Retake the image when the Daily page changes noticeably.
- Use `frontend/scripts/capture-static-site-tour.sh` or `frontend/scripts/capture-static-site-tour.mjs` when screenshots/GIFs need refreshing.
- Keep captions explicit that the static site is a reduced-functionality demo.
- If a live-only feature appears in the README, do not imply it exists in static mode unless the static export has a corresponding read-only view.
- RRG-capable markets carry compact `rrg-history-<market>.json.gz` weekly
  `static-rrg-history-v4` Group-strength history on the dedicated
  `rrg-history-data` release. Each successful Market build replaces its one
  asset; a missing asset is rebuilt, while a schema/formula-mismatched fallback
  is rejected instead of being merged into the active history.
- Breadth-enabled markets carry compact
  `breadth-contributor-metadata-<market>.json.gz` state on the dedicated
  `breadth-contributor-metadata-data` release. Keep state publication ahead of
  market-artifact upload so a published drilldown always has its retained
  company and IBD-group metadata available to the next build.

## Related Docs

- [Live App Guide](LIVE_APP_GUIDE.md)
- [Operations Guide](OPERATIONS.md)
- [Docker Deployment](INSTALL_DOCKER.md)
