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

Scheduled and normal manual Static Site workflows select the balanced formula independently for every Market in their fresh private build databases before Feature, Group, and RRG generation. The workflow's `rs_formula_overrides` JSON input exists only for coordinated per-Market rollback; omitted Markets remain balanced.

Balanced activation validates every declared Scan chunk and requires exact formula, Market RS run, RS date, and universe-size metadata on the Scan manifest, chunks, and rows. Its approval fingerprint covers the root manifest and complete `markets/<market>` tree, rather than only `manifest.json`.

`markets/<market>/groups.json` is an exact export of the stored Group snapshot for that formula/run. It includes Group overall, 1M, and 3M RS, with Group Rank based only on overall RS. The static Group table exposes separate **1M RS** and **3M RS** columns, matching the live table.

RRG rolling-history assets use `static-rrg-history-v4` and identify their RS formula version. The weekly RS-Ratio/RS-Momentum transformation is unchanged, but history from different formulas is never merged. A fallback market artifact or RRG-history bundle is rejected when its schema or formula does not match the requested export, so the site cannot silently combine legacy and balanced rankings. Until enough balanced history exists, RRG can correctly report insufficient history rather than falling back to legacy coordinates.

## Breadth Data Contract

Breadth-capable static market artifacts must contain a revision-3 `breadth.json` payload and a `breadth-r3` source marker. The combiner rejects revision-1 or revision-2 breadth artifacts even when their Market RS formula is otherwise compatible. If no current revision-3 artifact exists, that market's breadth is unavailable; the publisher does not silently reuse an older breadth formula.

Revision 3 uses the same fixed local-market StockBee thresholds as the live app and has no historical-FX input. Markets whose catalog capability disables breadth (currently AU, SG, and MY) publish their other supported static sections without requiring breadth or breadth-derived exposure.

Contributor drilldowns are optional additive assets. When available, a market entry advertises `assets.breadth_contributors.index_path`; the index uses `breadth-contributors-v1` and references at most 20 newest-first per-session documents under `markets/<market>/breadth/contributors/`. Each stock is stored once with all qualifying signals plus its frozen company name and IBD group. The combiner validates every document and reconciles its 11 supported signal counts to `breadth.json`. Missing legacy assets remain valid, while an invalid contributor asset is omitted without removing the market or its aggregate breadth page.

Static builds preserve contributor display metadata in the dedicated `breadth-contributor-metadata-data` release. Each breadth-enabled market owns a canonical `breadth-contributor-metadata-<market>.json.gz` asset containing only company names and IBD groups for its newest 20 contributor sessions, plus a `.previous` recovery asset that protects interrupted replacements. On the first run, all retained sessions are bootstrapped from the current enriched universe and classification data. Later runs restore that state, keep existing date/symbol metadata frozen, and bootstrap only newly encountered sessions or symbols. Restored bytes are validated for gzip, schema, and market identity. The new state must pass non-empty name and group coverage checks and publish successfully before the corresponding market artifact can replace its last-good fallback. A missing canonical asset uses the validated recovery asset when present; only the absence of both is a safe first publication. An uncertain restore or failed state upload suppresses the current market artifact.

The static frontend loads the small index with the market bundle and fetches a date document only after a user clicks an advertised nonzero Recent History count. The live and static dialogs therefore use the same schema and do not recalculate breadth formulas in the browser.

This rolling release is a static-build continuity mechanism only. The live app continues to read contributor metadata from its own PostgreSQL snapshots and does not depend on GitHub Releases.

## Maintaining Static Assets

- Keep the static page tour GIF aligned with the exported static routes.
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
