# Market history archive and `what_changed`

Agents can read today's snapshot but cannot ask what changed since yesterday. The site publishes only the newest session, and past fundamentals, ratings and screen matches cannot be rebuilt later. Retained artifacts expire after 30 days, so each day without an archive loses a day for good.

Two PRs, in this order. The archive ships first because it starts the clock. The tool ships once real archive files exist to test against.

1. **Archive.** Record every published session per market.
2. **Tool.** `what_changed` in `stockscreen-mcp`, released as 0.4.0.

## Storage decision

| Option | Result |
|---|---|
| Immutable per-session Release assets, one Release per market | Chosen. No file is ever rewritten, so a crash or a retry cannot damage history. Readers fetch two small files |
| Append to one file, like the scoreboard ledger | US would reach about 500 MB a year, and an agent would download all of it to compare two days |
| Pages | Every deploy would re-upload the window, and the publisher is stateless |
| A git data branch | Bloats every clone |

Measured on the 2026-11-06 publish with heavy fields dropped: 10.5 MB a day across 12 markets, 2.1 MB for US, about 2.6 GB a year. Keeping every field costs little and cannot be undone later, so the archive keeps full rows rather than a chosen subset.

A Release holds at most 1,000 assets. One Release per market gives about four years per tag, and `v1` in the tag leaves room for the next one.

## Data shape

Release tag `market-history-<m>-v1`, for example `market-history-us-v1`, created as a prerelease with `--latest=false`.

Asset `<m>-<YYYY-MM-DD>.json.gz`, gzip with `mtime=0`:

```json
{
  "schema_version": "market-history-v1",
  "market": "US",
  "as_of_date": "2026-11-06",
  "published_at": "2026-11-07T02:03:30Z",
  "origin": "live",
  "presets": {
    "minervini": {
      "name": "Minervini",
      "definition": "241a0f65f493c25e",
      "filter_expression": {},
      "symbols": ["AAA", "BBB"]
    }
  },
  "columns": ["symbol", "rs_rating"],
  "rows": [["AAA", 83.0]]
}
```

- `columns` is sorted and always holds `symbol`. A row missing a field holds `null` there.
- `rows` is sorted by symbol.
- `origin` reuses the ledger's `Origin` values, including `artifact_pre_stage_fix`.
- `definition` is the ledger's `definition_hash`, so a rule change is visible to readers.
- Rows drop only `price_sparkline_data`, `rs_sparkline_data` and `field_availability`, the fields `stockscreen_mcp.data.HEAVY_FIELDS` drops.

Asset `index.json`, rebuilt from the Release listing:

```json
{"schema_version": "market-history-index-v1", "market": "US", "sessions": ["2026-11-05", "2026-11-06"]}
```

## Rules

- **The first publication of a session wins.** A session asset is never overwritten, so `published_at` and the content mean what the site first showed. A later correction to the same session does not appear.
- **Any absent date can be added**, including one older than the newest date, because each session is its own file. An earlier session that a market's publish falls back to is recorded the same way.
- **The index is derived, never authoritative.** Every run compares it with the listing and rewrites it when they differ, so a crash between the session upload and the index upload heals on the next run.
- **Failures are per market.** One market going red does not stop the others, and the archive never affects the deploy or the ledger.
- **Mid-session behavior.** The archive reads the built site only. It fetches no prices and calls no provider, so a session means whatever the site published for that `as_of_date`.

## Pipeline

- `build` job: `python3 -m app.scripts.market_history extract --site-dir ../frontend/dist/static-data --output /tmp/market-history-days`, then upload artifact `market-history-days` with short retention. Both steps are `continue-on-error`, like the ledger's. Extraction reads the built site, so the rows are what deploys.
- New `record-history` job after `deploy`, `contents: write`, stdlib and `gh` only, using `needs.deploy.outputs.deployed_at`.
- `backfill --market <M>` reads retained default-branch artifacts oldest first. The first artifact per `as_of_date` wins, with `origin` from `artifact_origin` and `published_at` from `created_at`. It runs from a dispatch-only workflow, `market-history-backfill.yml`. It does not share the publisher's concurrency group: session assets are immutable, the index heals itself, and sharing the group would let a backfill replace a pending publish.

Reuse from `scoreboard_ledger.py` rather than copying: `Origin`, `artifact_origin`, `check_publishable`, `definition_hash`, `site_readers`, `match_presets`, `default_branch_artifacts`, `download_artifact`, `format_timestamp`, `parse_timestamp`, `run_gh`, `MARKETS`. `upload_asset` is bound to the ledger's tag, so give it a tag parameter that defaults to the ledger's.

## Tool, for the second PR

`what_changed` in `stockscreen-mcp`, released as 0.4.0. It reads `index.json` and two session assets from `https://github.com/xang1234/stock-screener/releases/download/market-history-<m>-v1/`, overridable with `STOCKSCREEN_HISTORY_URL`. Session assets are gzip files served as plain bytes, not with `Content-Encoding: gzip`, so the reader gunzips by content.

```
what_changed(market="US", since=None, until=None, symbols=None, screen=None, limit=25)
```

- **Dates.** `until` picks the newest archived session on or before it, default the newest. `since` picks the newest archived session before `until` and on or before `since`, default the session just before `until`. A date with no session on or before it is an error naming the first archived session. Fewer than two sessions is an error saying when the archive started.
- **Screens.** For each preset in either session: `entered` and `left` counts and up to `limit` symbols each, sorted by symbol, with a `truncated` flag. A symbol whose row is absent on one side is reported under `new_to_publish` or `dropped_from_publish`, never as entering or leaving. `definition_changed` is true when the two definitions differ, because then a move can come from the rule rather than the stock. `screen` narrows to one preset.
- **Symbols.** Up to 50. For each: present on each side, the screens it entered and left, and each tracked field whose value changed, as `from` and `to`. Tracked fields are a fixed table in the code, and a test checks each is in `contracts/scan_field_dictionary.json`.
- **Context, always returned.** Both dates with `published_at` and `origin`, how many archived sessions lie between them, `rs_universe_size` and `rs_formula_version` on each side, and `warnings`. A warning is raised when the RS universe size differs by more than 10 percent, when the RS formula differs, when either side is `artifact` or `artifact_pre_stage_fix`, when the two sides straddle 2026-10-21 (`artifact_pre_stage_fix` era), when a definition changed, and when the newest publish is newer than the newest archived session.
- **how_to_read.** One fixed note, like `screen_track_record`'s, saying moves can come from data coverage, not only from the stocks, and that prices are not split-adjusted across sessions.
- **Limits.** Responses over `MAX_RESPONSE_CHARS` are refused with a hint, as in `screen_track_record`.
- **Caching.** Session files never change, so they are cached by market and date; keep at most four in memory. The index is fetched with the manifest's TTL. A 404 on `index.json` is reported as the archive being briefly unavailable, because a `--clobber` replace deletes before it uploads.
