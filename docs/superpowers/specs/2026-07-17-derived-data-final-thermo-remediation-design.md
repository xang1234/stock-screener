# Derived-data final thermo remediation

## Context

Issue 301 now keeps breadth and group-ranking calculations cache-only after the
daily price-refresh guard, while preserving provider-capable manual historical
runs. The implementation passes its behavioral suite, but the final strict
review found four structural risks:

1. optimized group backfill deletes and commits an entire date range before it
   proves replacement rankings can be calculated;
2. gap-fill tasks invoke decorated daily Celery tasks in-process through task
   introspection and `ContextVar` bypasses;
3. group taxonomy ownership is split between the facade and loader, and the
   loader reads every group's membership twice;
4. the supposedly typed group-ranking result still carries mutable
   `Mapping[str, Any]` payloads into persistence.

The remediation must preserve the already-approved behavior: public Celery
task names and arguments, legacy request-flag precedence, legacy prefetch
adapter support, serialized response fields, manual provider fallback, and
refresh-guarded provider exclusion.

## Goals

- Preserve existing rows whenever a historical date cannot be recalculated.
- Make replacement of one date's rankings atomic.
- Stop invoking decorated Celery tasks from other Celery tasks.
- Make the group input loader the single owner of production taxonomy loading.
- Fetch each group's constituent membership once per input load.
- Carry an explicit immutable ranking type from calculation through
  persistence.
- Delete private duplicate ranking implementations after parity is protected
  by public calculator tests.
- Keep all external compatibility contracts unchanged.

## Non-goals

- Removing legacy Celery keyword arguments.
- Removing `LegacyGroupRankPrefetchAdapter`.
- Changing strict, automatic, or refresh-guarded completeness thresholds.
- Making manual historical jobs cache-only.
- Unifying breadth and group ranking behind a generic derived-data framework.
- Making an entire multi-date backfill one database transaction.
- Changing API schemas or the stored `ibd_group_ranks` table.

## Chosen approach

Use focused structural remediation rather than a generic coordinator. Group
ranking receives a typed calculation/persistence path and atomic per-date
replacement. Breadth and group tasks receive Celery-free daily runners that
both direct and gap-fill task adapters call. Compatibility remains at the
existing public task and legacy-adapter boundaries.

This is preferred to a minimal patch because a minimal patch would leave the
in-process Celery invocation and arbitrary ranking dictionaries intact. It is
preferred to a shared generic framework because breadth and group ranking have
different validation, persistence, and diagnostic contracts; forcing them
behind one abstraction would add modes rather than delete complexity.

## Typed group-ranking contract

Add a frozen `GroupRanking` data model to `group_rank_models.py` with these
fields:

- `industry_group: str`
- `date: date`
- `rank: int`
- `avg_rs_rating: float`
- `median_rs_rating: float | None`
- `weighted_avg_rs_rating: float | None`
- `rs_std_dev: float | None`
- `num_stocks: int`
- `num_stocks_rs_above_80: int`
- `top_symbol: str | None`
- `top_rs_rating: float | None`

`GroupRankCalculationResult.rankings` becomes `tuple[GroupRanking, ...]`.
The calculator may use a private immutable candidate type before ranks are
known; after sorting it constructs final `GroupRanking` values rather than
mutating dictionaries. `GroupRankingRepository.store_rankings()` accepts only
`Sequence[GroupRanking]` and maps attributes to database columns. Serialization
to dictionaries happens only where an external response or compatibility test
requires it.

The production-only legacy calculation helpers are removed. Parity remains
covered through fixtures and assertions against `calculate_for_date()` and
`calculate_for_dates()`, not through a second production implementation.

## Single-pass group inputs

`GroupRankPrefetchData` gains an ordered `group_names: tuple[str, ...]` field.
The production loader performs these steps:

1. resolve the market and load group names once;
2. resolve the benchmark according to the execution policy;
3. load the active universe once;
4. load each group's raw constituent tuple once;
5. partition that tuple into active supported symbols and active unsupported
   symbols in the same loop;
6. derive `symbols_by_group`, unique price targets, and unsupported counts from
   that one captured taxonomy snapshot;
7. batch-load prices and market caps.

The group-ranking facade no longer loads groups before calling the loader. It
uses `prefetch.group_names`, raises `MissingIBDIndustryMappingsError` when that
tuple is empty, and passes the same names to the calculator.

The legacy adapter sets `group_names=()` because the five-item legacy tuple did
not carry group names. The explicit legacy-completion seam may load group names
and memberships once when a legacy tuple is supplied. That exceptional work
remains isolated in the adapter/facade compatibility path and is never used by
the production typed loader.

Benchmark-missing results retain their group names so the caller can
distinguish missing taxonomy from missing benchmark data without performing a
second taxonomy query.

## Atomic historical replacement

Remove delete-before-rebuild from optimized backfill. Calculation always
precedes mutation.

Add a repository operation with the conceptual contract:

```python
replace_rankings_for_date(
    db,
    *,
    calculation_date: date,
    rankings: Sequence[GroupRanking],
    market: str,
) -> int
```

For one date it deletes the existing market-scoped rows and stores all
replacement rows without committing. The historical calculator then commits
that date. If delete, insert, or commit fails, it rolls back; the previous rows
for that date remain intact. Dates that produce no rankings are counted as
errors and are not mutated.

Optimized backfill still prefetches once and calculates in configured chunks.
It then replaces successful dates independently. This provides failure
isolation without holding locks for the entire range. The returned `deleted`
count is the sum of rows replaced by successfully committed dates.

Manual backfill tasks bump the group-ranking epoch and publish snapshots only
when at least one date was successfully replaced. If no date was processed,
they return the existing error/statistics response without advertising a new
snapshot. Partial success is safe to publish because failed dates retain their
previous complete rows.

## Celery-free daily runners

Create focused daily runner modules for breadth and group ranking. A runner is
a normal callable, not a Celery task, and accepts resolved Python values such
as `date`, normalized market, resolved `DerivedDataExecutionPolicy`, a database
session, and its service dependencies.

The runners own daily domain work:

- execute the calculation service;
- apply the domain-specific completeness validation;
- persist the daily result;
- perform existing cache-epoch and UI-snapshot publication side effects;
- return a typed internal outcome that task adapters serialize to the existing
  dictionaries.

The public direct Celery tasks continue to own transport concerns:

- parse serialized dates and legacy arguments;
- resolve the execution policy once;
- open and close sessions;
- publish task/activity lifecycle state;
- translate domain exceptions into the established reason codes and response
  keys;
- schedule Celery retries for transient failures.

The gap-fill Celery tasks call the same daily runners directly for the target
date. They do not call `.run()`, inspect `__module__`, detect mocks, bypass the
workload decorator, or use a `ContextVar` to change retry propagation. A
transient runner exception naturally reaches the calling task adapter, so the
outermost Celery task schedules the only retry.

The existing same-day warmup-bypass context managers remain because static
feature export uses them as an external compatibility seam. Only the nested
task invocation and transient-propagation bypasses are removed.

## Error handling and transactions

- Missing group taxonomy remains a named failure with its current reason code.
- Missing benchmark or insufficient cache coverage retains current strict and
  tolerant behavior.
- A daily runner never schedules a Celery retry; it raises and lets its task
  adapter decide.
- Failed historical calculation does not delete existing rows.
- Failed historical persistence rolls back only the affected date.
- UI snapshot publication remains best-effort and must not roll back committed
  ranking data.
- No-success manual backfills do not bump epochs or publish snapshots.

## Compatibility boundaries

The following remain stable:

- Celery task registration names;
- arguments including `force_cache_only`,
  `refresh_guarded_cache_only`, and `execution_policy`;
- legacy precedence where strict cache-only wins;
- public task response keys and reason codes;
- refresh-guarded cache-only provider exclusion;
- provider-capable manual historical defaults;
- `LegacyGroupRankPrefetchAdapter` support for five-item tuples;
- database schema and API payloads.

Internal private helper names, arbitrary ranking dictionaries, and test-only
legacy calculation methods are not compatibility contracts and will be
removed.

## Testing strategy

Implementation follows red-green-refactor cycles.

### Historical safety

- A missing benchmark performs no delete and no commit.
- A date producing no rankings preserves its previous rows.
- A store failure rolls back the date and preserves its previous rows.
- Successful replacement deletes and inserts in one transaction.
- A zero-success task does not bump the epoch or publish a snapshot.
- Partial success publishes only after successful date commits.

### Typed rankings

- Calculators return immutable `GroupRanking` values with correct rank order.
- Repository tests accept the typed model and preserve PostgreSQL/fallback
  behavior.
- Static checks reject `Mapping[str, Any]` ranking contracts in the calculator,
  result, and repository.
- Public task outputs retain their current scalar fields.

### Input ownership

- Production input loading calls `groups()` exactly once.
- `symbols_for_group()` is called exactly once per group.
- Supported and unsupported counts derive from the captured membership tuple.
- The facade does not call taxonomy before typed loading.
- Legacy tuple completion still resolves missing group metadata.

### Orchestration

- Direct tasks resolve compatibility flags and call the daily runner once.
- Gap-fill tasks call the runner rather than the decorated task.
- Transient failures cause only the outer Celery task to retry.
- Existing direct, guarded, strict, same-day, historical, and static-export
  behavior tests remain green.
- Source checks reject task-object `.run()` invocation and the removed transient
  propagation `ContextVar` in breadth/group task modules.

### Verification

Run focused group model, loader, calculator, repository, historical, task, and
breadth suites during development. Before completion, run the comprehensive
backend unit suite, backend quality gates, frontend suite when shared contracts
change, `git diff --check`, and confirm the pull-request checks after pushing.

## Acceptance criteria

- Historical recalculation failure cannot erase previously committed rankings.
- Each successful date replacement is atomic.
- Production group taxonomy membership is loaded once per group per run.
- Group rankings cross calculation and persistence as immutable typed values.
- Breadth/group gap-fill does not invoke decorated Celery daily tasks in-process.
- Public compatibility and issue-301 provider-exclusion behavior remain intact.
- Focused and comprehensive verification passes.

