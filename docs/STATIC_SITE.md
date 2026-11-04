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

A market entry can advertise `assets.scoreboard = {"path": "markets/<m>/scoreboard.json"}`, a small file of what happened after each preset's published matches. It adds no schema bump: the path validator checks it like any other asset, and one unparseable advertised file rejects the whole market.

Past matches cannot be rebuilt, because fundamentals keep no history. So the `scoreboard-ledger-data` release stores what the site published, one line per first-published date, in `scoreboard-ledger-<m>-v1.jsonl.gz` with `scoreboard-ledger-<m>-v1.previous.jsonl.gz` as the prior copy. The ledger stores symbols and a hash of each preset's `filter_expression`. It never stores returns or entry prices, because every build recomputes them from current prices.

Two workflows touch the ledger:

- The `record-ledger` job in `static-site-publish.yml` runs after `deploy`, with the same `if`, so it records what actually deployed. It holds `contents: write`, installs nothing and runs `backend/app/scripts/scoreboard_ledger.py`, which is stdlib only. A date at or before the ledger's end uploads nothing.
- The `Build market scoreboard` step in `static-site.yml` downloads the ledger and runs `backend/app/scripts/build_static_scoreboard.py` as its own process, with `continue-on-error` and a 20 minute cap. This workflow never writes the release. If the step fails, `manifest.market.json` stays unchanged and the market publishes without the asset.

Returns follow these rules:

- **Entry and exit.** Entry is the open of the first session that opens after publication. Exit is the close of the Nth session, for N of 5, 20 and 60. Returns are in local currency, price only, with no costs.
- **Prices.** The builder reads `stock_prices` directly, capped at the export's `as_of_date`, so it cannot see a partial bar from a session in progress and cannot reach a provider. A recent IPO still scores, because there is no 50-bar minimum.
- **Overlap.** A repeat of a symbol inside its counted window is suppressed. The rule runs per preset across the preset's whole history, so a definition or era change does not reset it, and a symbol in two presets counts once in each.
- **Cohorts.** Numbers are never pooled across `(preset, definition, origin)`. `origin` is `live`, `artifact` (read from a retained build artifact, whose `created_at` only approximates publication) or `artifact_pre_stage_fix` (the same, before the 2026-10-22 stage fix).
- **Delisted stocks.** A symbol with no price counts in `skipped_missing_history`. Stocks that later delist drop out this way, which can flatter results, so the count is published beside the returns.

Each horizon block has `sample_size`, `signal_dates`, `pending_windows`, `oldest_open_sessions`, `thin`, `positive_rate` (a fraction from 0 to 1), `avg_return_pct`, `median_return_pct`, `avg_excess_return_pct`, `median_excess_return_pct`, `excess_sample_size`, `skipped_missing_history` and `suppressed_overlapping`. `thin` means fewer than 20 closed events or fewer than 5 signal dates. `oldest_open_sessions` is null when no window is open. Statistics with no closed window are null.

`missing_sessions` counts completed sessions between `recording_since` and `ledger_through` with no ledger line. To seed the release, recover a lost day or repair a corrupt asset, see the [scoreboard ledger runbook](runbooks/scoreboard-ledger.md).

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
