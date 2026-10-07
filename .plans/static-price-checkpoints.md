# Static price-stage checkpoints (#502)

## Problem

A static market export refreshes raw OHLCV for the whole market, then runs
derived work (feature snapshot, Market RS, breadth, groups, scan, export).
The runner's PostgreSQL is ephemeral, so when the job hits its
`timeout-minutes` every fetched price is lost and the next attempt starts
from the last published daily bundle again.

Observed (US, run 37548280301, 2026-10-06): job 79 min; price stage
23:56 to 00:35 (39 min, 8,210 stale + 1,844 bootstrap symbols); derived work
about 30 min. A throttled Yahoo stretches the price stage, and that is when
the 300-minute cap is reached.

## Design (lean)

A deadline checkpoints the price stage before the runner times out. A re-run
of the same session imports the checkpoint and fetches only what is still
missing.

1. **Deadline.** `StaticDailyPriceRefreshService` takes a monotonic
   `deadline` (and an injectable `clock`). Before each provider batch, and
   before the rate-limited retry wait and the latest-session repair wait,
   it checks the time left. Once the deadline has passed it starts no further fetch and returns
   `status: "resumable"`. Rows from finished batches are already committed:
   `store_batch_in_cache(also_store_db=True)` commits each batch.
2. **Checkpoint write.** `export_static_site --price-stage-deadline-minutes N
   --price-checkpoint-dir D` runs the refresh with that deadline. On
   `resumable` it skips all derived work, exports the market's rows with the
   existing `export_daily_price_bundle` to `D/price-checkpoint-<m>.json.gz`,
   writes the manifest `D/price-checkpoint-<m>.json` (with
   `kind: "price_checkpoint"` and a fingerprint), and exits `80`.
3. **Upload.** The workflow uploads data first and the manifest last to
   `daily-price-data`, three attempts each, with `--clobber`, then fails the
   job with a "re-run failed jobs to resume" error. There is one fixed name
   per market, so assets never accumulate. A crash between the two uploads
   leaves a manifest whose `sha256` no longer matches the data, which is
   rejected.
4. **Resume.** `python -m app.scripts.resume_static_price_checkpoint --market
   M` runs after the daily-bundle seed. It downloads the checkpoint through
   `GitHubReleaseSyncService.fetch_latest_bundle`, which verifies the
   SHA-256. It rejects a manifest whose `kind` or fingerprint differs from
   the one computed now, then imports in **checkpoint mode**:
   - each symbol's rows from its first checkpoint date onward are replaced
     inside one transaction, because `persist_stock_price_mappings` only
     updates a symbol's latest row and old-scale history would otherwise
     survive a split replacement;
   - the completed daily-import state (`github_sync.daily_prices.<m>`) is
     not advanced.

   The refresh then reclassifies coverage, so finished symbols count as
   fresh and only the remainder is fetched.
5. **Fingerprint.** Fields: checkpoint format; market; as-of session; bundle
   schema and bar period; baseline daily-bundle revision (the seeded import
   state); weekly-reference revision; price provider plan (version and
   ordered providers); adjustment-drift tolerance; refresh periods. Any
   mismatch means incompatible: the checkpoint is ignored and the run takes
   the normal path.
6. **Opt-in.** The repo variable `STATIC_PRICE_STAGE_DEADLINE_MINUTES` is a
   JSON object of market to minutes, e.g. `{"US": 210}`. Markets not listed
   behave exactly as today, with no resume and no checkpoint. Rollback:
   delete the variable. Leftover assets are non-authoritative and need no
   cleanup.

### What is deliberately not built (vs. the issue)

- **No separate `--mode repair` lane.** Adjustment-drift and short-history
  symbols that miss the deadline stay stale or short in the database, so the
  resumed refresh classifies them again and continues their bounded
  replacement. Coverage reclassification is the "recorded repair scope".
- **No `describe` CLI.** The resume script prints `missing`, `incompatible`,
  `invalid` or `imported`.
- **No completed-symbol list.** Coverage reclassification is the
  deterministic equivalent.
- **No per-request HTTP timeout capping.** The deadline sits about 90 minutes
  before the job timeout, which absorbs one in-flight batch (about 30 s
  normally, minutes under backoff).
- **No new timing reporter.** The refresh result reports elapsed seconds,
  and the existing batch log lines carry timestamps.
- **No periodic mid-stage uploads.** A hard kill (runner loss, external
  cancel) still loses the stage. That is the same as today, and is #503's
  domain.
- **No automatic re-dispatch.** The failed job is the signal; "Re-run failed
  jobs" re-runs only that market on the same session.

A checkpoint never produces `daily-price-latest-<m>.json`, a market
artifact, or a publisher wake-up. Only a later complete export does.

## Tasks

1. Refresh deadline: `deadline`/`clock` on the service; checks before
   batches and waits; `resumable` result. Tests: a stop after N batches
   stores exactly N batches; no fetch after the deadline; the retry wait and
   session-repair wait are not started past the deadline.
2. Checkpoint module `app/services/static_price_checkpoint.py`: names,
   fingerprint, write, resume. Add `checkpoint=True` to
   `DailyPriceBundleService.import_daily_price_bundle` (window replacement,
   no import state). Tests: write then resume round-trip on SQLite; each
   fingerprint field mismatch is rejected; a normal latest manifest under
   the checkpoint name is rejected; checksum mismatch; import state and
   latest manifest unchanged; rollback on a bad row; old-scale rows are
   replaced; resuming twice is idempotent.
3. Export wiring: flags, early return in `_run_daily_refresh`, checkpoint
   write in `main`, exit `80`. Script `resume_static_price_checkpoint`.
   Tests: export returns 80 and writes the checkpoint without derived work.
4. Workflow and docs: job env from the variable; resume step; export
   arguments and status 80; upload step with retries (data then manifest);
   fail step. Workflow tests and `docs/OPERATIONS.md`.
