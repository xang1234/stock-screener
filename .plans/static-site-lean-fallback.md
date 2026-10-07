# Static site: visible, valid last-good fallback (#498, revised)

Issue: #498 (scope revised 2026-10-06; the Release-backed catalog was dropped).
Parent: #497.

## Goal

A static market never silently disappears, and a valid earlier artifact is
published when the current export fails. Data inputs are already durable in
Releases, so recovery from a long outage is "fix the producer, dispatch the
market group"; no durable output store is built.

Artifact retention stays at the repository default (30 days).

## Changes

### 1. Fallback policy (`backend/app/scripts/validate_static_market_artifacts.py`)

`_ALLOWED_FALLBACK_REASONS` gains `export_failed`. A current-run export failure
describes the new candidate, not the history. `has_current_artifact: true`
still blocks a fallback-backed omission (unchanged).

### 2. Unavailable markets and publication state

**Combiner** (`backend/app/services/static_artifact_combiner.py`):
`_build_manifest` adds

```json
"unavailable_markets": ["IN"]
```

listing supported markets (in `supported_markets` order of the combiner's
configured list) with no selected artifact. `supported_markets` and `markets`
keep meaning "served" so existing readers are unchanged. Omitted market trees
are still removed: an unavailable market advertises no data path.

**Publication block** (`StaticSiteExportService.combine_market_artifacts`):
each served `markets[M]` entry gains

```json
"publication": {
  "source": "current" | "fallback",
  "session_date": "<entry.as_of_date>",
  "session_lag": 0,
  "state": "current" | "stale" | "unknown"
}
```

`session_lag` = `market_session_lag(calendar, market, start_date=session_date,
end_date=calendar.last_completed_trading_day(market))`; `state` is `current`
at 0, `stale` above 0, `unknown` (with `session_lag: null`) if the calendar
raises or a date is missing. The combiner records each selected artifact's
`source_label`; the service computes lag after combine with
`MarketCalendarService()` (no DB, as `report_static_market_freshness` already
does). Calendar is injectable for tests. Lag is computed at combine time; a
mid-session run sees the previous completed session as "current", which
matches the freshness gate.

**Frontend** (`frontend/src/static/dataClient.js`, `StaticLayout.jsx`):
`getStaticUnavailableMarkets(manifest)` returns the array (or `[]`). The
market `<Select>` renders them after served markets as disabled `MenuItem`s
labelled `<CODE> — unavailable`. No page changes: an unselectable market
never resolves to a page. A stored/query `?market=IN` already normalizes to
the default market because `supported_markets` excludes it.

### 3. Direct artifact lookup (`backend/app/scripts/download_static_market_fallbacks.py`)

Replace the workflow-run scan with, per artifact name (`static-market-<M>` for
every supported market, plus the options and COT names):

`gh api --paginate --slurp repos/{repo}/actions/artifacts?name=<name>&per_page=100`

Candidate filter (trust boundary):
- not `expired`;
- `workflow_run.id != current_run_id`;
- `workflow_run.head_branch == branch`;
- `workflow_run.head_repository_id == workflow_run.repository_id`
  (excludes fork-PR runs whose head branch happens to share the name).

Order by `created_at` descending. Keep the existing per-candidate download
(`gh run download <run_id> --name`), compatibility/asset/formula checks and
atomic install. Keep the best `as_of_date` seen; stop when
`_run_cannot_beat_incumbent(created_at date, incumbent)`. An invalid
candidate is skipped, so an older valid artifact is used instead.

Removed: `extract_runs`, the runs API call, the per-run artifact listing.

### 4. Advertised-path validation (combiner `_validate_advertised_assets`)

Add one helper `_validate_advertised_paths(market, source_label, entry,
market_dir)` called first in `_validate_advertised_assets` (used by both the
downloader and `_discover`):

- every `entry.pages[*].path` and every `entry.assets[*].path` (dict
  descriptors with a `path`), after stripping the `markets/<m>/` prefix, must
  resolve inside `market_dir`, exist, and parse as JSON;
- the chart index (`assets.charts.path`) entries' `path`s must resolve inside
  `market_dir` and exist (not parsed: there can be hundreds).

Failure raises `StaticArtifactFormulaError` (the existing rejection type).
The contributor asset keeps its own soft-drop validation.

## Mid-session behaviour

Nothing here reads or fetches prices. `session_lag` uses
`last_completed_trading_day`, so a market whose session is open is judged
against the previous close.

## Tests (failing first)

- `tests/unit/test_static_market_artifact_validation.py`: `export_failed`
  with a fallback now passes; flip the existing rejection case.
- `tests/unit/services/test_static_artifact_combiner.py`:
  `unavailable_markets` lists an omitted optional market and its tree is
  removed; a missing/unparsable page path and a chart payload outside the root
  are rejected; `source` recorded per selected artifact.
- `tests/unit/services/` (static site export service): `publication` block
  with a fake calendar for lag 0, lag > 0 and a raising calendar.
- `tests/unit/test_download_static_market_fallbacks.py` (exists or new):
  by-name lookup selects the newest valid date, skips expired/current-run/
  other-branch/fork-repo artifacts, falls through an invalid newest artifact,
  stops early once `created_at` cannot beat the incumbent.
- `frontend/src/static`: selector shows a disabled unavailable market.

```bash
(cd backend && ../../../../backend/venv/bin/python -m pytest tests/unit/test_static_market_artifact_validation.py tests/unit/services/test_static_artifact_combiner.py tests/unit/test_static_site_workflow.py -q)
(cd frontend && npm run test:run -- src/static)
```

## Out of scope

Release snapshots, catalog ref, CAS, accepted/deployed state, cleanup,
retention change, page-level stale UI (#503).
