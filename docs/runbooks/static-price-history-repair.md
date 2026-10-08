# Static Price History Repair (RS anchors)

Canonical stock RS needs a usable adjusted close at eight exact sessions: the as-of session and the 1, 5, 21, 63, 126, 189 and 252-session anchors. A symbol missing any of them gets no RS rating and no composite score. A hole in old history therefore stays hidden until a horizon's anchor rolls onto it. In #539, AU and DE lost RS for most of their universe on 2026-10-06, when the 21-session anchor moved onto 2026-09-07.

## What every static build now does

`StaticDailyPriceRefreshService` checks each RS market's active symbols against the anchors of the as-of session and of the next 10 sessions (`RS_ANCHOR_LOOKAHEAD_SESSIONS`). The check does not depend on `group_rankings`.

- **What counts as a gap:** the symbol has stored history older than the anchor but no usable adjusted close on it. A symbol whose history starts after the anchor (a new listing) is a short history, not a gap, and is not refetched.
- **Gaps more than 4 calendar days back:** the symbol is refetched for 2 years. Its stored rows are replaced only if the refetch covers every stored date in range plus the missing anchors, so two adjustment bases are never spliced. Otherwise nothing changes and the symbol stays unresolved.
- **Gaps in the last 4 days:** these go to the normal 7-day top-up.
- **Recheck:** coverage is rechecked from committed rows. A successful fetch is not counted as a repair.

The result is `price_refresh.rs_anchor_repair`, also printed in the job log:

| Field | Meaning |
|---|---|
| `gap_symbols`, `gap_count_by_date` | Gaps found before the repair, per session |
| `repaired_symbols` | Gaps that committed rows now cover |
| `unresolved_symbols`, `unresolved_count_by_date`, `unresolved_samples` | Gaps still open, e.g. the provider lacks the session or returned an error |
| `status` | `verified`, or `unverified` with `error` when the calendar could not resolve anchors |

The job log lines start with `[static-daily prices:<MARKET>]`, for example `RS anchor repair: 1695/1695 repaired, 0 unresolved {}`.

### Publication guard

The Market RS input loader rejects a run when more than 25% of currently priced symbols miss an anchor despite older history (`MAX_HISTORY_ANCHOR_GAP_SHARE`). The reason code is `historical_adjusted_anchor_gap_above_threshold`, and its diagnostics give per-anchor counts and samples.

The market then follows the existing `market_rs_not_ready` path: no current artifact is published, and the publisher serves the last good one. The daily-price bundle is still built and uploaded (`has_price_bundle` stays true), so rows repaired in that run carry into the next seed. Short histories and the current-session coverage gate are unchanged.

## Operator repair for one market

A normal run repairs every symbol whose current or next-10-session anchors are missing; each repair refetches 2 years, which also fills that symbol's other holes. To check every session the 252-session horizon can anchor on, dispatch a single-market repair:

```bash
gh workflow run static-site.yml \
  --repo xang1234/stock-screener \
  --ref main \
  -f market_group=asia \
  -f repair_price_history_market=AU
```

Pick the `market_group` that contains the market (AU is in `asia`, DE in `us`). Only the named market's job gets `--repair-price-history`. Repair one market first, verify it, then repair the next.

To run the same thing locally against a database seeded from the release bundles:

```bash
cd backend
python -m app.scripts.export_static_market_artifact \
  --output-dir /tmp/static-data --refresh-daily --build-mode price_delta \
  --skip-universe-refresh --skip-fundamentals-refresh --skip-cot-refresh \
  --market AU --repair-price-history
```

Do not use `backend/scripts/force_full_cache_refresh.py`: it deletes every price row.

## Verify

1. **Repair:** in the market job's log, check the `RS anchor repair` line: `repaired_symbols`, `unresolved_symbols`, and the per-date counts for the incident range.
2. **Bundle:** check that the job uploaded `daily-price-<market>-YYYYMMDD.json.gz` and then `daily-price-latest-<market>.json` to the `daily-price-data` release. The bundle is uploaded before its manifest.
3. **Clean import:** in a clean database, import the new bundle with `python -m app.scripts.import_daily_price_bundle`. Then confirm the eight anchors are present for previously affected symbols.
4. **RS and scores:** compare the market's RS-eligible count and the scored count in the deployed scan with the pre-incident baseline. Check them separately: RS eligibility does not imply a composite score.
5. **Next ordinary run:** check that it imports the repaired bundle and reports `gap_symbols` near zero, so the collapse does not return as anchors advance.

A green workflow, a local repair or a bundle upload alone is not production recovery. The fix is recovered only with clean-import and deployed-result evidence.

## Rollback

- **Bundle:** `release-asset-cleanup` keeps the 10 newest dated bundles per market. To roll back, re-upload a manifest that points at an earlier `daily-price-<market>-YYYYMMDD.json.gz`, using `gh release upload daily-price-data <manifest> --clobber`.
- **Site:** the publisher keeps serving the last good market artifact while a market has no current artifact.
- **Concurrency:** static-site runs share one queued concurrency group (`queue: max`), so an older run cannot overwrite a newer repaired bundle concurrently.
