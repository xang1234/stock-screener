# Per-market Load/Soak Harness (bead asia.9.3)

Measures the per-market rate-budget plumbing from beads 9.1 + 9.2 under
realistic concurrency. Produces a JSON snapshot per run, compares against a
committed baseline, and fails CI when wall-clock or 429 incident counts
regress beyond thresholds.

## What it tests

The canonical scenario is the Sunday weekly refresh window, when 4 per-market
beat entries (`weekly-full-refresh-{us,hk,jp,tw}`) fire simultaneously. The
harness:

1. Spawns one `ThreadPoolExecutor` worker per supported market.
2. Each worker runs `BulkDataFetcher.fetch_prices_in_batches(market="...")`
   against a deterministic yfinance simulator.
3. The real `RateBudgetPolicy` + `RedisRateLimiter` + per-market lock keys
   from 9.1 + 9.2 govern concurrency end-to-end.
4. After all workers finish, the harness writes wall-clock, 429 count,
   tail-latency, and resource samples to a JSON snapshot.

## How to run

```bash
# Standard run (gates against the committed baseline)
make gate-6-load

# Update the baseline after an intentional change
make load-baseline-update

# Run pytest directly
cd backend
pytest tests/load/ -v -m load
```

Requires a running Redis instance (the rate-limiter atomic ops and 429
counters need real Redis; mocking would defeat the purpose). When Redis is
unavailable the suite is skipped rather than failing.

## Live yfinance evidence-pack runs

The default harness uses a deterministic simulator for CI stability. To
produce real-traffic numbers for an evidence pack:

```bash
LOAD_TEST_LIVE=1 pytest tests/load/ -v -m load
```

⚠️ This is currently a stub — the live runner needs additional safeguards
(IP rate limiting, longer timeouts, separate baseline). Track in a future
follow-up if real-traffic measurements become a regular need.

## Regression thresholds

Defined in `measurement.py`:

| Metric             | Threshold | Gated? |
|--------------------|-----------|--------|
| `wall_clock_s`     | +20%      | yes    |
| `rate_limit_429s`  | +50%      | yes    |
| Tail-latency p95   | (n/a)     | reported only |
| Worker memory/CPU  | (n/a)     | reported only |

The two gated metrics fail the `gate-6-load` target when they regress beyond
the threshold for any market. Tail-latency and resource samples are written
to the snapshot for human review but do not fail the build — they're for
post-hoc trend analysis when wall-clock regressions need diagnosis.

## Updating the baseline

When a change is expected to shift performance (e.g. tuning a per-market
rate constant after evidence shows the original was wrong), regenerate the
baseline:

```bash
make load-baseline-update
git add backend/tests/load/baselines/per_market_load.json
git commit -m "perf(load): update load baseline after <change>"
```

The baseline JSON is committed to the repo so `git diff` shows perf changes
per commit. A regression-blocking PR can be unblocked by re-running with
`LOAD_TEST_UPDATE_BASELINE=1` and committing the new baseline alongside the
change that caused it.

## Synthetic universe

For reproducibility the harness uses fixed per-market symbol counts:

| Market | Symbols |
|--------|---------|
| US     | 1000    |
| HK     | 500     |
| JP     | 300     |
| TW     | 200     |

Defined in `conftest.py:SYNTHETIC_UNIVERSE_SIZES`. The numbers are
proportional to real production universe sizes but small enough that the
harness completes in seconds rather than minutes.

## Fault Isolation (bead asia.9.4)

Beyond performance regression detection, the harness includes chaos-lite
fault-injection tests that validate the operational guarantee 9.1 + 9.2
were designed to deliver: **a failure in one market is contained to that
market**.

### How it works

`fault_injection.py` exposes three helpers that mutate a single market's
behavior, then `test_failure_isolation.py` runs the parallel 4-market
refresh and asserts the **non-affected** markets behave identically to the
healthy baseline (`per_market_load.json`).

For each non-affected market, the chaos tests assert:

1. Symbol processing count matches the healthy baseline (no incomplete refresh).
2. Transient-failure count matches the healthy baseline (no failures leak from victim).
3. Per-market `ratelimit:429:yfinance:<market>` Redis counter is 0 (rate-key isolation).
4. `yfinance_calls` matches the healthy baseline exactly (deterministic batch behavior unchanged).

All four assertions are CI-gated. The affected (victim) market is allowed
to fail or partially succeed — only the requirement that the harness itself
doesn't crash is enforced for the victim.

### Fault scenarios → isolation mechanism mapping

| Test | Fault injected | Isolation mechanism validated |
|---|---|---|
| `test_provider_exhaustion_isolation` | Victim returns 100% yfinance 429s | Per-market `yfinance:<market>` rate-limit keys (9.2) |
| `test_provider_hard_failure_isolation` | Victim raises non-429 exception every call | `BulkDataFetcher` shared-state safety + per-market batch loops |
| `test_lock_stuck_isolation` | Victim's `data_fetch_job_lock:<market>` held externally | Per-market lock keys (9.1) |

### Run

```bash
make gate-7-chaos

# Or directly
pytest backend/tests/load/test_failure_isolation.py -v -m load
```

The chaos tests share `tests/load/`'s simulator, harness, and Redis-reset
fixture with `gate-6-load`. They can be run independently or as part of the
nightly load+chaos suite.

## Why this isn't gated per-PR

Load tests are inherently slower and more side-effect-heavy than unit tests
(real Redis, threading, time-based measurements). Running on every PR adds
CI time and produces flaky failures from CI-host noise. The recommended
schedule is:

- **Nightly**: scheduled GitHub Actions workflow runs `make gate-6-load`
  and reports regressions.
- **Manual**: developers run `make gate-6-load` locally before merging
  changes that touch `services/rate_budget_policy.py`,
  `services/rate_limiter.py`, `tasks/data_fetch_lock.py`, or
  `services/bulk_data_fetcher.py`.
