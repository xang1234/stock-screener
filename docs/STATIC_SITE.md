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

## Maintaining Static Assets

- Keep the static page tour GIF aligned with the exported static routes.
- Use `frontend/scripts/capture-static-site-tour.sh` or `frontend/scripts/capture-static-site-tour.mjs` when screenshots/GIFs need refreshing.
- Keep captions explicit that the static site is a reduced-functionality demo.
- If a live-only feature appears in the README, do not imply it exists in static mode unless the static export has a corresponding read-only view.
- RRG-capable markets carry compact `rrg-history-<market>.json.gz` weekly
  group-strength history on the dedicated `rrg-history-data` release. Each
  successful market build replaces its one asset; a missing or invalid asset
  is rebuilt from the two-year price bundle and static group-rank bootstrap.

## Related Docs

- [Live App Guide](LIVE_APP_GUIDE.md)
- [Operations Guide](OPERATIONS.md)
- [Docker Deployment](INSTALL_DOCKER.md)
