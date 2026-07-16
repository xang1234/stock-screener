# Refresh-guarded cache-only breadth and group calculations

**Date:** 2026-07-16
**Status:** Approved in conversation; written-spec review pending
**Issue:** GitHub #301 / Beads `stockscreenclaude-duw`

## Problem

The daily market pipeline refreshes persisted prices and accepts the refresh when
at least 90% of the active universe covers the target trading date. It then runs
breadth and group-ranking tasks for that explicit date.

Those downstream tasks interpret an explicit date before the market's current
date as a historical/backfill calculation. Historical breadth and group paths
currently permit live provider fallback, even though the preceding refresh
stage has already established the price state that the pipeline accepted. The
automatic gap-fill work inside both stages also permits provider fallback.

The result is unbounded work after the refresh guard. Breadth can issue serial
per-symbol provider calls, while group ranking can make a bulk provider request.
For the run documented in GitHub #301, breadth saw 439 initial cache misses and
435 insufficient-history outcomes, indicating that most fallback work did not
produce calculation-ready data.

The three stages do not currently share one meaning of completeness:

- The price guard checks target-date presence and accepts 90% coverage.
- Breadth needs enough history for returns through 63 trading days. Its strict
  cache-only mode rejects cache-miss ratios above 10%.
- Group ranking needs stock and benchmark history for relative strength. Its
  strict cache-only mode requires 95% US stock coverage.

Reusing the existing strict `force_cache_only` behavior in the guarded pipeline
would therefore risk converting previously tolerated symbol gaps into new
pipeline failures.

## Goal

After the daily price-refresh guard succeeds, all breadth and group work for the
pipeline's explicit target date and automatic gap-fill dates must use persisted
price history only. Missing, stale, or insufficient histories are skipped and
reported. They must not cause a provider call or, by themselves, fail the
stage.

## Non-goals

- Changing the price-refresh guard's 90% acceptance threshold.
- Changing strict cache-only behavior used by static exports or manual callers.
- Changing explicit manual/backfill commands that have not opted into guarded
  cache-only behavior.
- Recovering permanently unavailable securities inside calculation tasks.
- Changing breadth formulas, relative-strength formulas, or publication dates.
- Addressing the separate feature-snapshot orphan/stale-run lifecycle.

## Chosen design

### Explicit guarded policy

Add a task-level boolean named `refresh_guarded_cache_only`, defaulting to
`False`, to the breadth and group gap-fill wrappers and their nested daily task
entry points. The daily pipeline passes it as `True` only after the price guard.

The flag is distinct from `force_cache_only`:

- `force_cache_only=True` retains today's strict static/manual completeness
  behavior.
- `refresh_guarded_cache_only=True` disables provider access but tolerates and
  reports per-symbol gaps.
- If both flags are true, strict `force_cache_only` behavior wins.
- With both flags false, existing same-day and historical behavior is unchanged.

This explicit policy is preferred over treating every explicit date as
cache-only because manual historical backfills may intentionally recover data
from providers. It is preferred over new pipeline-specific Celery tasks because
the existing wrappers already own date resolution, gap-fill, activity, and
error handling.

### Pipeline flow

The guarded daily flow becomes:

1. Run `smart_refresh_cache(mode="delta")`.
2. Require `guard_price_refresh` to succeed.
3. Run breadth gap-fill with `refresh_guarded_cache_only=True`.
4. Run target-date breadth with the same policy.
5. Run group gap-fill with `refresh_guarded_cache_only=True`.
6. Run target-date group ranking with the same policy.
7. Continue to the feature snapshot when the existing breadth/group result
   guards accept the outputs.

Celery immutable signatures remain in use, so downstream tasks do not consume
or reinterpret the guard's result payload. The ordering of the chain is the
authorization: guarded cache-only tasks cannot start unless the price guard has
returned successfully.

### Breadth behavior

`BreadthCalculatorService.fill_gaps()` gains a `cache_only` argument and passes
it to `backfill_range()`. The wrapper passes `cache_only=True` for guarded runs.
The target-date task passes `cache_only=True` to
`calculate_daily_breadth()`.

On a guarded run:

- `get_many_cached_only_fresh()` is the only stock-history read path.
- `_calculate_stock_history()` and `get_historical_data()` are never called.
- Cache misses and insufficient histories increment existing skipped/miss
  counters.
- Gap-fill returns cache-miss counts and a bounded symbol sample.
- The strict 10% cache-miss rejection is not applied. A non-empty usable breadth
  population is published with degradation diagnostics.

Actual database/calculation exceptions and a zero usable breadth population
remain failures. Individual unavailable symbols are not failures.

### Group-ranking behavior

`IBDGroupRankService.fill_gaps_optimized()` gains a `cache_only` argument and
passes it to `_prefetch_all_data()`. The group wrapper uses it for gap-fill and
propagates `refresh_guarded_cache_only` to the nested target-date task.

On a guarded run:

- Stock histories come from `get_many_cached_only_fresh()`.
- Benchmark history comes from cached-only benchmark candidates.
- `price_cache.get_many()`, benchmark provider fallback, and all other provider
  paths are excluded.
- The strict 95% group cache requirement is not applied.
- Groups are ranked from usable constituents, retaining the existing minimum
  constituent requirements.
- Prefetch statistics, cache misses, and a bounded missing-symbol sample are
  included in task results for observability.

A missing benchmark or zero rankable groups remains a hard failure because no
meaningful ranking can be produced. Missing individual constituents do not fail
the stage.

### Result and error contract

Guarded tasks return `cache_only: true` and
`cache_policy: "refresh_guarded"`. Successful partial calculations remain
successful task results rather than using the existing `partial` status, which
the pipeline guards interpret as failure.

Diagnostics include, where applicable:

- total candidate symbols;
- symbols with usable histories;
- cache-miss count;
- insufficient-history/skipped count;
- cache coverage ratio;
- a bounded sample of missing symbols; and
- gap-fill dates processed or errored.

No provider retry is scheduled from a guarded breadth/group task. Transient
database and infrastructure exceptions retain existing retry/error behavior.

## Safety and compatibility

The change is opt-in at the daily pipeline signature. Default values preserve
all existing direct callers. Static export and manual strict cache-only callers
retain their existing coverage gates. Manual historical backfills retain live
fallback unless they explicitly choose the guarded policy.

The daily snapshot remains cache-only as it is today. This change only removes
provider access from the derived-data stages that precede it.

## Testing

Tests are added at each boundary where provider access could leak:

1. Pipeline signature tests prove breadth and group wrappers receive
   `refresh_guarded_cache_only=True` only after the price guard.
2. Breadth wrapper tests prove the flag reaches both `fill_gaps(cache_only=True)`
   and the nested target-date task.
3. Breadth service tests prove cache misses never call
   `_calculate_stock_history()` and that usable partial data succeeds with
   diagnostics.
4. Group wrapper tests prove the flag reaches both optimized gap-fill and the
   nested target-date task.
5. Group service tests prove guarded prefetch uses cached-only stock and
   benchmark reads and never calls provider-capable methods.
6. Partial-coverage tests prove missing individual histories do not return an
   error when a usable breadth population or rankable groups remain.
7. Compatibility tests prove strict `force_cache_only` coverage failures and
   default manual historical provider fallback remain unchanged.

Focused regression coverage is run for daily pipeline, breadth task/service,
and group task/service tests, followed by the backend unit suite permitted by
the repository's current test baseline.
