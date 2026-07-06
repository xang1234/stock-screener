# Local Daily Snapshot Date Coherence

## Goal

Keep local live API payloads and exported static payloads coherent for a
market's completed trading day, regardless of the operator's machine timezone.

The concrete bug was a Hong Kong static snapshot whose scan data represented
one market date while live sections such as market exposure, breadth, groups,
and RRG could drift to the latest available local data.

## Date Contract

- The canonical dashboard anchor is the scan feature run's `as_of_date`.
- The anchor is a market trading date, not the browser date, server UTC date, or
  scan `completed_at` date.
- Daily snapshot sections that can drift forward must query at that anchor:
  market breadth, market exposure, key markets, top groups, and group RRG.
- `scan.completed_at` may be shown as publication freshness when no feature run
  exists, but it must not pin section queries.
- When no canonical anchor exists, the snapshot is explicitly `unanchored` and
  sections may use their latest available data.
- Scheduled local tasks resolve "today" through the target market calendar.
  Explicit `calculation_date` values are preserved and passed through to nested
  daily calculations.

## Implementation Notes

- `daily_snapshot_service._snapshot_anchor_date()` returns only
  `scan.feature_run.as_of_date`.
- The daily snapshot builder passes that anchor to breadth, exposure, key
  markets, and groups.
- Group RRG endpoints already accept `as_of_date`; the live groups page now
  forwards the fresh bootstrap ranking date into the RRG bundle request.
- Breadth and group-ranking task wrappers share `resolve_task_target_date()` so
  explicit dates and market-local scheduled dates are handled consistently.
- The freshness payload includes `snapshot_as_of_date`, section latest dates,
  `market_timezone`, and `date_coherence_status` for operator visibility.

## Verification

Primary regression checks:

- Backend daily snapshot tests prove anchored sections are pinned to the scan
  feature-run date.
- Backend daily snapshot tests prove scans without a feature run do not use
  `completed_at` as a query anchor.
- Task date-resolution tests prove explicit dates bypass market "today" lookup,
  while scheduled runs use the target market calendar.
- Frontend group page tests prove live RRG fallback passes the bootstrap ranking
  date as `as_of_date`.

Manual timezone check:

- Run the local site under different host timezones.
- Select a market whose completed trading date differs from the host calendar
  date.
- Confirm Daily Snapshot freshness shows the same `snapshot_as_of_date` and all
  pinned sections use that same market date.
