# PR 303 Review Remediation Design

**Status:** Approved on 2026-07-18

**Goal:** Address the still-valid review findings on PR #303 without breaking
existing task contracts, manual workflows, historical provider behavior, or
the cache-only refresh guard.

## Scope

This remediation covers current runtime, data-integrity, compatibility,
documentation, and CI behavior identified by the PR review. Historical
implementation-plan documents are records of completed work and must not be
rewritten in response to review comments about their prior state.

The change must preserve:

- existing Celery task names and arguments;
- existing task response payloads for non-transient failures;
- provider-capable manual historical breadth and group runs by default;
- cache-only execution after the refresh guard;
- the legacy `BreadthCalculatorService.backfill_range(..., cache_only=True)`
  call shape;
- optional construction of the daily snapshot use case without an injected
  calendar for US compatibility.

## Review Disposition

| Finding | Disposition |
|---|---|
| US snapshot trading-day helper always raises | Fix; restore a real canonical US calendar check. |
| Static breadth export passes `cache_only=True` | Fix with a compatibility shim on `backfill_range`. |
| Cache-only breadth accepts histories without the target-session bar | Fix at the cache-only read boundary for daily calculations. |
| Cache-only group ranking accepts histories without the target-session bar | Fix at the shared cache-only read boundary for daily calculations. |
| Breadth coverage can classify a symbol as both cached and missing | Fix with strict batch membership and conflict validation. |
| Non-positive group chunk size can loop forever | Fix with validation before iteration. |
| Group backfill tasks swallow transient database failures | Fix by propagating only recognized transient database errors. |
| Strict cache-only breadth can publish with zero usable stocks | Fix by rejecting zero-scanned results. |
| Breadth calculation docstring still describes a dictionary | Fix the return documentation. |
| Manual refresh header access assumes a dictionary | Fix with guarded mapping access. |
| CI has redundant collection and does not exclude opt-in markers in sharded collection | Fix the sharded collection command and remove the redundant step. |
| Group taxonomy membership is loaded twice | No code change; already fixed and regression-tested on the current branch. |
| Full backend suite is known to have 54 failures | No code change; finding is outdated and all four required CI shards pass. |
| Historical implementation-plan comments | No document changes; reply that the file is an immutable execution record. |
| Previously resolved atomic replacement, market normalization, and cache invalidation comments | No code change; retain their resolved state. |

## Design

### Target-session cache validation

`PriceCacheService.get_cached_only_fresh` and
`PriceCacheService.get_many_cached_only_fresh` will accept an optional
`required_as_of_date: date | None` keyword argument. Existing calls omit it and
retain current freshness behavior. When supplied, the returned DataFrame must
contain a row whose normalized index date equals the required date. A globally
fresh frame that lacks that session is returned as a cache miss.

Daily breadth passes its calculation date to the bulk cache-only read. Daily
group ranking passes its calculation date through
`IBDGroupRankService._prefetch_all_data` to `GroupRankInputLoader.load`; the
loader applies the required date to both cached benchmark selection and cached
constituent reads. Provider-allowed and multi-date historical prefetch paths do
not supply the new argument and retain their existing behavior.

This boundary makes coverage diagnostics authoritative: a symbol that cannot
support the requested session contributes to cache-miss statistics instead of
being ranked or scanned from a prior close.

### Breadth completeness and compatibility

Strict cache-only validation will reject a result when
`total_stocks_scanned == 0`, even if cached histories were present but all were
insufficient. Refresh-guarded validation already enforces this condition and
is unchanged.

`BreadthPriceCoverageAccumulator.record_batch` will reject missing symbols
that are not in the supplied candidate batch. It will also reject a repeated
symbol whose cached/missing classification conflicts with an earlier batch.
The accumulator will remain unchanged after a rejected batch, ensuring its
report always has disjoint cached and missing sets drawn only from candidates.

`BreadthCalculatorService.backfill_range` will regain an optional
`cache_only: bool | None = None` keyword. When omitted, `policy` remains
authoritative. When supplied, it maps to the equivalent historical
provider-allowed or strict-cache-only policy. This preserves legacy callers
without weakening the typed policy used by current tasks.

The daily breadth docstring will describe `BreadthCalculationResult`, including
`indicators`, `coverage`, policy-derived metadata at the task boundary, and
`to_metrics_dict()`.

### Group backfill safety

`backfill_rankings_chunked` will raise `ValueError` before entering its loop
when `chunk_size_days < 1`. Positive values retain the existing iteration and
statistics behavior.

The three administrative group backfill task handlers will continue rolling
back and logging all failures. Before returning their legacy error dictionary,
they will call the existing transient-database classifier and re-raise only a
recognized transient database exception. The surrounding
`serialized_market_workload` decorator will then invoke its established Celery
retry path. Non-transient exceptions retain the existing response payload.

### Snapshot calendar and refresh header safety

The `_is_us_trading_day` compatibility helper will delegate to the canonical
market calendar for `US` rather than raising unconditionally. The injected
calendar remains preferred inside `BuildDailyFeatureSnapshotUseCase`; direct
US construction and the task-level compatibility guard become functional
again.

Manual full-refresh detection will read request headers once and require a
mapping before calling `.get("origin")`. The recognized value remains exactly
`"manual"`.

### CI behavior

The isolated shard-1 safe-collection step will be removed because each shard
already performs collection. The actual node-id collection used for sharding
will include `-m "not live_service and not load"`, so opt-in tests cannot enter
the required backend unit job even if they are misplaced under `tests/unit`.
All four shards remain required dependencies for image publication.

## Error Handling

- Missing target-session rows are normal cache misses and use existing
  coverage diagnostics.
- Invalid breadth coverage batches raise `ValueError` because they indicate an
  internal accounting contract violation.
- Invalid group chunk sizes raise `ValueError` synchronously.
- Only SQLAlchemy database errors classified as transient escape administrative
  task handlers for retry; other errors preserve legacy return values.
- Calendar lookup failures remain visible rather than being misreported as a
  non-trading day.

## Test Strategy

Every behavioral fix follows a separate red-green cycle:

1. Prove the US task guard no longer aborts every US snapshot.
2. Prove `cache_only=True` remains callable and provider-free.
3. Prove cache-only breadth treats a prior-session frame as a miss.
4. Prove cache-only group inputs treat stale benchmark and constituent frames
   as unavailable for the requested session.
5. Prove breadth coverage rejects unknown misses and conflicting repeated
   classifications without corrupting prior state.
6. Prove strict cache-only breadth rejects zero usable stocks.
7. Prove non-positive group chunk sizes fail immediately.
8. Prove transient database errors reach the workload retry wrapper while
   non-transient failures retain their dictionaries.
9. Prove non-mapping task headers do not raise and mapping headers preserve
   manual behavior.
10. Parse CI YAML and assert the sharded collection command contains the marker
    exclusion while the redundant step is absent.

After focused tests pass, run the complete breadth/group task suites, snapshot
and static-export suites, the complete backend unit suite, generated-contract
checks, repository quality gates, and `git diff --check`. Push the branch and
wait for all PR checks before resolving review threads.

## GitHub Review Responses

Valid inline findings will be answered in their existing GitHub threads after
the corresponding fix and verification are pushed. Outside-diff findings will
receive one concise PR comment listing their disposition. Historical-plan
threads will be resolved with the explicit decision that completed plans are
not rewritten. Already-fixed and outdated findings will be resolved with
current code or CI evidence.

## Non-Goals

- Rewriting historical implementation plans.
- Raising repository-wide docstring coverage to CodeRabbit's generic target.
- Changing cache-coverage thresholds.
- Changing manual historical provider access.
- Changing task names, public arguments, or response schemas beyond restoring
  the documented compatibility parameter.
- Refactoring unrelated price-refresh, breadth, group-ranking, or CI code.
