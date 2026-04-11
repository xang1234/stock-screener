# ASIA v2 E2 Migration Rehearsal Report (ST3 + T2 Validation)

- Date: 2026-04-11
- Beads:
  - `StockScreenClaude-asia.2.1.3` (ST3 Run migration rehearsal and rollback drill)
  - `StockScreenClaude-asia.2.2` (T2 Add market identity fields and backfill US baseline)
- Scope: Alembic revisions `20260410_0002` -> `20260411_0003` -> `20260411_0004`

## Objective

Validate migration safety and rollback viability on a production-like dataset before launch-gate usage in `asia.11.*`, with explicit evidence for:

1. schema upgrade correctness (`0003`, `0004`);
2. US-baseline backfill correctness for new identity fields;
3. rollback and re-upgrade operability.

## Environment and Dataset

- Database: isolated local PostgreSQL 16 container (`asia-rehearsal-pg`), DB `stockscanner_asia_rehearsal`.
- Runtime: `uv run --python 3.11 --with-requirements requirements-server.txt`.
- Seeded rows before `0004`: 5,006 total.
  - 5,000 US-style symbols (`US000001` ... `US005000`).
  - 6 representative non-US formats: `0700.HK`, `9988.HK`, `7203.T`, `6758.T`, `2330.TW`, `3008.TWO`.

## Execution Log (Timed)

| Step | Command Outcome | Real Time |
|---|---|---|
| Upgrade to `20260410_0002` | Success | 24.67s |
| Upgrade to `20260411_0003` | Success | 2.48s |
| Downgrade `0004` -> `0003` (rollback drill) | Success | 1.55s |
| Re-upgrade `0003` -> `0004` (post-seed) | Success | 1.16s |

Notes:
- Additional downgrade/re-upgrade cycles were executed during drill iterations (`~1.05s` and `~1.10s`) with no migration errors.
- Full command evidence is captured in `/tmp/asia_rehearsal.log` during rehearsal execution.

## Backfill Validation (`asia.2.2`)

After upgrading to `20260411_0004` on the seeded dataset:

- `rows_total = 5006`
- `backfill_metrics = (0, 0, 0, 0, 6, 6, 6, 6)` on the mixed-market fixture where tuple fields are:
  - `market_missing`
  - `currency_missing`
  - `timezone_missing`
  - `local_code_missing`
  - `non_us_market_rows`
  - `non_us_currency_rows`
  - `non_us_timezone_rows`
  - `local_code_not_symbol_rows`

Sample spot-check rows:

- `0700.HK` -> `(market='HK', currency='HKD', timezone='Asia/Hong_Kong', local_code='0700')`
- `2330.TW` -> `(market='TW', currency='TWD', timezone='Asia/Taipei', local_code='2330')`
- `7203.T` -> `(market='JP', currency='JPY', timezone='Asia/Tokyo', local_code='7203')`

Interpretation:
- Existing rows were backfilled deterministically using exchange/suffix inference with US fallback.
- `local_code` derives exchange-local identifiers for suffixed non-US symbols and falls back to `symbol` otherwise.

## Rollback Viability (`asia.2.1.3`)

Rollback assertion executed:

1. Downgrade from `20260411_0004` to `20260411_0003`.
2. Validate removal of identity columns from `stock_universe`.
3. Re-upgrade to `20260411_0004`.

Observed rollback check:

- `has_market_identity_columns = False` at revision `0003`.
- Re-upgrade to `0004` succeeds without manual intervention.

Conclusion: rollback path is operational and re-entrant for this migration pair.

## Blockers and Resolutions

1. `uv run` panic under sandbox with default runtime:
   - Symptom: Tokio/system-configuration panic.
   - Resolution: run outside sandbox and pin Python to 3.11.
2. Python 3.13 dependency build failure (`pydantic-core`):
   - Symptom: `ahash`/`stdsimd` compile error.
   - Resolution: `uv run --python 3.11 ...`.
3. Initial seed script violated `consecutive_fetch_failures` NOT NULL:
   - Resolution: include explicit `consecutive_fetch_failures = 0` in seed inserts.

## Acceptance Mapping

- `StockScreenClaude-asia.2.1.3`: Met.
  - Upgrade/downgrade rehearsal executed.
  - Timing and blocker evidence recorded.
  - Rollback viability demonstrated.
- `StockScreenClaude-asia.2.2`: Met.
  - Migration applied.
  - Backfill validated on representative seeded dataset.
