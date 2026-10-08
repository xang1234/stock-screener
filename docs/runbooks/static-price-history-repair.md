# Static Price History Repair (RS anchors)

Canonical stock RS needs a usable adjusted close at eight exact sessions: the as-of session and the 1, 5, 21, 63, 126, 189 and 252-session anchors. A symbol missing any of them gets no RS rating and no composite score. A hole in old history therefore stays hidden until a horizon's anchor rolls onto it. In #539, AU and DE lost RS for most of their universe on 2026-10-06, when the 21-session anchor moved onto 2026-09-07.

## What every static build now does

`StaticDailyPriceRefreshService` checks each market's active symbols and benchmarks against the anchors of the as-of session and of the next 10 sessions (`RS_ANCHOR_LOOKAHEAD_SESSIONS`). The check does not depend on `group_rankings`.

- **What counts as a gap:** a missing anchor that falls inside the symbol's stored history (after its first row, up to its last). New listings and dormant or halted symbols are not gaps and are not refetched.
- **Gaps more than 4 calendar days back:** the symbol is refetched for 2 years, with one retry after a rate-limit failure.
  - Its stored rows from the refetch's first date onward are replaced only if the refetch covers every one of them plus the missing anchors, so two adjustment bases are never spliced. A truncated refetch leaves the symbol unresolved.
  - Gaps that an earlier fetch in the same run already filled are rechecked from stored rows and not refetched.
  - Otherwise nothing changes and the symbol stays unresolved.
  - The repair runs before the latest-session quote repair.
- **Gaps in the last 4 days:** these go to the normal 7-day top-up.
- **Recheck:** coverage is rechecked from committed rows. A successful fetch is not counted as a repair.

The result is `price_refresh.rs_anchor_repair`, also printed in the job log:

| Field | Meaning |
|---|---|
| `gap_symbols`, `gap_count_by_date` | Gaps found before the repair, per session |
| `repaired_symbols` | Gaps that committed rows now cover |
| `unresolved_symbols`, `unresolved_count_by_date`, `unresolved_samples` | Gaps still open, e.g. the provider lacks the session or returned an error |
| `status` | `verified`, or `unverified` with `error` when the calendar could not resolve anchors |

The job log lines start with `[static-daily prices:<MARKET>]`, for example `RS anchor repair: 1695/1706 repaired, 11 unresolved {...}`.

### Measured baseline (2026-10-07 daily-price bundles)

| Market | Priced symbols missing an anchor despite older history | Symbols the lookahead check refetches |
|---|---|---|
| AU | 92% (the incident) | 1,706 |
| DE | 16% (about 137 sparse illiquid names) | 449 |
| CA, HK, IN, JP, KR, MY, SG, TW | under 1% | 0–66 |

DE's sparse names are probably refetched on every run and stay unresolved, because the provider has no bars for their no-trade days. That is expected noise, and the run remains bounded at about 450 symbols.

### Publication guard

The Market RS input loader records `history_gaps` on each run: the share of currently priced symbols missing an interior anchor despite older history, plus per-anchor counts and samples. The live app only records it, so live RS stays partial rather than failing.

The static export enforces it. When `history_gaps.share` is above 50% (`STATIC_RS_MAX_HISTORY_GAP_SHARE`), the market's RS is rejected with `historical_adjusted_anchor_gap_above_threshold`. The market then follows the existing `market_rs_not_ready` path: no current artifact is published, and the publisher serves the last good one. The daily-price bundle is still built and uploaded, so rows repaired in that run carry into the next seed.

## Operator repair for one market

A normal run already repairs every symbol whose current or next-10-session anchors are missing, with a 2-year refetch that also fills that symbol's other holes. To check every session the 252-session horizon can anchor on, dispatch a single-market repair from `main`:

```bash
gh workflow run static-site.yml --repo xang1234/stock-screener --ref main \
  -f market_group=asia -f repair_price_history_market=AU
gh workflow run static-site.yml --repo xang1234/stock-screener --ref main \
  -f market_group=us -f repair_price_history_market=DE
```

Choose the group that contains the market (AU is in `asia`, DE in `us`). The run fails early if the market is not in the selected group. Only the named market's job gets `--repair-price-history`. Repair one market, verify it, then repair the next.

Do not use `backend/scripts/force_full_cache_refresh.py`: it deletes every price row.

### If the provider genuinely lacks the sessions

The repair log then shows the same dates under `unresolved_count_by_date` on every run, and the guard keeps the market on its last good artifact. It will freeze again as each longer horizon (63, 126, 189, 252 sessions) reaches the hole.

To publish anyway with the reduced RS universe, set the repository variable and pass it to the export step's environment:

```bash
gh variable set STATIC_RS_MAX_HISTORY_GAP_SHARE --repo xang1234/stock-screener --body 1
```

Record why, then remove the variable once the hole has aged out of the 252-session window.

## Verify

1. **Repair:** in the market job's log, check the `RS anchor repair` line: repaired and unresolved counts, and the per-date counts for the incident range.
2. **Bundle:** check that `daily-price-<market>-YYYYMMDD.json.gz` and then `daily-price-latest-<market>.json` were uploaded to the `daily-price-data` release. The bundle is uploaded before its manifest.
3. **Clean import:** import the bundle into an empty database (`DATABASE_URL` pointing at it, `alembic upgrade head` applied) and check the anchors:

   ```bash
   cd backend
   gh release download daily-price-data --repo xang1234/stock-screener \
     --pattern 'daily-price-latest-au.json' --pattern 'daily-price-au-*.json.gz' --dir /tmp/dp
   ASSET="$(python -c "import json;print(json.load(open('/tmp/dp/daily-price-latest-au.json'))['bundle_asset_name'])")"
   python -m app.scripts.import_daily_price_bundle --input "/tmp/dp/$ASSET" \
     --manifest /tmp/dp/daily-price-latest-au.json --expected-market AU \
     --expected-bundle-asset-name "$ASSET"
   python - <<'PY'
   import json
   from datetime import date
   from app.database import SessionLocal
   from app.models.stock import StockPrice
   from app.services.market_calendar_service import MarketCalendarService
   from app.services.rs_anchor_price_coverage import RsAnchorPriceCoverageService

   as_of = date.fromisoformat(json.load(open("/tmp/dp/daily-price-latest-au.json"))["as_of_date"])
   svc = RsAnchorPriceCoverageService(calendar_service=MarketCalendarService())
   with SessionLocal() as db:
       symbols = [s for (s,) in db.query(StockPrice.symbol).distinct()]
       required = svc.required_dates(market="AU", through_date=as_of, lookahead_sessions=0)
       gaps = svc.gaps(db, symbols=symbols, required_dates=required)
   print(len(gaps.missing_dates_by_symbol), "symbols with anchor gaps", gaps.count_by_date())
   PY
   ```

   (The download pattern matches every retained AU bundle. The import uses the one the manifest names.)
4. **RS and scores:** compare the deployed scan's RS-eligible count and scored count with the pre-incident baseline. Check them separately: RS eligibility does not imply a composite score.
5. **Next ordinary run:** check that it imports the repaired bundle and reports few `gap_symbols`, so the collapse does not return as anchors advance.

A green workflow, a local repair or a bundle upload alone is not production recovery. The fix is recovered only with clean-import and deployed-result evidence.

## Rollback

- **Bundle:** `release-asset-cleanup` keeps the 10 newest dated bundles per market. To roll back, re-upload a manifest that points at an earlier `daily-price-<market>-YYYYMMDD.json.gz`, using `gh release upload daily-price-data <manifest> --clobber`.
  - Bundles are named by as-of date and uploaded with `--clobber`, so a repair run on the same as-of date as the last ordinary run replaces that date's bundle.
  - To keep the pre-repair bundle, download it first: `gh release download daily-price-data --pattern 'daily-price-au-YYYYMMDD.json.gz'`.
- **Site:** the publisher keeps serving the last good market artifact while a market has no current artifact.
- **Concurrency:** static-site runs on the same ref share one queued concurrency group (`static-site-<ref>`, `queue: max`). Dispatch repairs from `main` so they queue behind, never beside, scheduled runs.
