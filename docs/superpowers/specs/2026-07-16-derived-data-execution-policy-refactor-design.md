# Derived-data execution policy and diagnostics refactor

**Date:** 2026-07-16
**Status:** Implemented and verified
**Issue:** GitHub #301 / Beads `stockscreenclaude-duw`
**Supersedes:** The implementation structure in
`2026-07-16-refresh-guarded-cache-only-derived-data-design.md`; its behavioral
requirements remain in force.

## Context

The refresh-guarded implementation correctly prevents breadth and group
historical calculations from reaching providers after the price-refresh guard.
The first implementation nevertheless introduced four maintainability defects:

1. The execution behavior is represented by combinations of
   `force_cache_only`, `refresh_guarded_cache_only`, same-day inference, and
   warmup-bypass context rather than one explicit policy.
2. Group ranking returns rankings normally but exposes prefetch diagnostics by
   mutating a caller-owned dictionary.
3. Group task and service modules, plus their tests, have crossed or moved
   further beyond useful size boundaries.
4. Breadth coverage is calculated by the service and then partially
   reconstructed by task code, producing multiple definitions and
   nondeterministic sampling across batches.

This refactor addresses those structural findings without changing the
provider-exclusion and partial-coverage behavior already approved for issue
#301.

## Goals

- Represent derived-data execution intent with one typed, serializable mode.
- Resolve execution behavior once at task boundaries and pass a typed policy
  through in-process service calls.
- Preserve all existing Celery argument compatibility, including already
  queued tasks.
- Return group rankings and typed prefetch diagnostics through one explicit
  result object.
- Give the breadth service sole ownership of cache-coverage accounting.
- Reduce oversized group production and test modules by moving cohesive
  responsibilities into focused modules.
- Preserve manual, bootstrap, static-export, and daily-pipeline behavior.

## Non-goals

- Changing the price-refresh guard or its 90% target-date threshold.
- Changing breadth formulas, group RS formulas, ranking thresholds, or
  publication behavior.
- Removing legacy Celery arguments in this change.
- Changing the behavior of manual historical runs that permit provider
  fallback.
- Solving the separate orphaned/stale feature-snapshot investigation.
- Introducing a general workflow framework or strategy-class hierarchy.

## Execution policy

### Serializable request mode

Add `DerivedDataExecutionMode`, a string enum with three values:

- `auto`
- `strict_cache_only`
- `refresh_guarded`

Celery tasks accept a new optional `execution_policy` string. Strings remain
safe to serialize across brokers and persisted signatures. Internal code parses
the string into the enum and rejects unknown values with the task's normal
error contract.

The existing public arguments remain accepted:

- `force_cache_only`
- `refresh_guarded_cache_only`

They are compatibility inputs only. No downstream task or service branches on
them after policy resolution.

### Compatibility precedence

Policy resolution is deterministic:

1. `force_cache_only=True` resolves to `strict_cache_only`.
2. Otherwise, `refresh_guarded_cache_only=True` resolves to
   `refresh_guarded`.
3. Otherwise, a supplied `execution_policy` is parsed and used.
4. Otherwise, the request resolves from `auto`.

This preserves the established rule that strict mode wins when both legacy
booleans are true. It also makes old queued calls behave exactly as before.

Wrappers forward only `execution_policy=<mode.value>` to nested tasks. The
daily pipeline constructs new signatures with
`execution_policy="refresh_guarded"`; it no longer emits the legacy guarded
boolean.

### Resolved in-process policy

Add an immutable `DerivedDataExecutionPolicy` with:

- requested `mode`;
- `cache_only`;
- `strict_completeness`;
- `requires_warmup_metadata`;
- `tolerates_partial_coverage`.

The resolver receives the requested mode, target date, market-local current
date, and the existing in-process warmup-bypass state:

| Requested mode | Target | Price access | Completeness | Warmup metadata |
|---|---|---|---|---|
| `auto` | historical | provider allowed | existing historical behavior | no |
| `auto` | same day | cache only | strict | yes |
| `strict_cache_only` | any | cache only | strict | bypassed as explicit static/manual intent |
| `refresh_guarded` | any | cache only | usable partial coverage | no |

The existing in-process warmup-bypass context remains supported for bootstrap
and static-export paths that enter through `auto`. It is read once while
resolving the policy; policy consumers see only the resolved
`requires_warmup_metadata` decision.

Services receive the resolved policy rather than the task-level legacy
booleans. Provider-capable versus cached-only reads are selected from
`policy.cache_only`; completeness validation is selected from the other policy
properties.

## Group ranking contracts

### Typed prefetch statistics

Create an immutable `GroupRankPrefetchStats` data class containing:

- target symbol count;
- symbols with usable prices;
- cache-miss count;
- deterministic, bounded cache-miss symbol tuple;
- coverage ratio;
- benchmark availability, symbol, and role;
- market and cache-only state;
- unsupported-symbol count;
- optional cache-requirement minimum and reason.

It exposes an explicit `to_dict()` method only for Celery/API result
serialization. Applying a strict cache requirement returns a replaced instance;
the service never mutates the statistics.

`GroupRankPrefetchData.stats` becomes `GroupRankPrefetchStats` instead of a raw
dictionary.

### Calculation result

Add `GroupRankCalculationResult` containing:

- `rankings`;
- `prefetch_stats`.

`IBDGroupRankService.calculate_group_rankings()` returns this result. The
`diagnostics` output parameter is removed. Every caller reads rankings and
statistics from the returned value, so success data and diagnostics share one
visible contract.

No-result paths still return an empty rankings collection with the authoritative
prefetch statistics. Cache-completeness exceptions carry the typed statistics
and serialize them explicitly where task responses require dictionaries.

### Input-loader extraction

Move group benchmark, universe, constituent, market-cap, and bulk-price
prefetching to `GroupRankInputLoader`. It owns:

- cached-only benchmark candidate selection;
- active-universe and group-symbol collection;
- unsupported-symbol exclusion;
- provider-capable versus cached-only bulk reads;
- construction of `GroupRankPrefetchData` and `GroupRankPrefetchStats`.

`IBDGroupRankService` remains responsible for RS calculation, group aggregation,
ranking, persistence, and historical ranking operations. Small delegation
methods may remain temporarily where they preserve internal test seams, but the
prefetch algorithm has one implementation in the loader.

## Breadth coverage contract

### Authoritative accumulator

Create a service-owned `BreadthCoverageAccumulator`. It records symbol identity
in sets and calculation outcomes in counters:

- candidate symbols;
- symbols with cached history;
- cache-miss symbols;
- successfully scanned observations;
- skipped observations;
- insufficient-history observations;
- error observations.

The accumulator produces an immutable `BreadthCoverageReport`. Missing-symbol
samples are sorted once and truncated to 20 when the report is created, making
the result independent of batch order. Coverage ratio is derived only by the
report.

The report provides explicit serializers for the existing daily-task and
backfill result field names. Compatibility aliases are produced at this
serialization boundary, not recalculated in task code.

### Service results

`BreadthCalculatorService.calculate_daily_breadth()` returns a typed
`BreadthCalculationResult` containing indicator metrics and a
`BreadthCoverageReport`.

Backfill/gap-fill methods use the same accumulator and return their existing
workflow statistics with the report's serialized coverage fields. Daily and
backfill paths therefore share one definition of candidate count, cache miss,
coverage, insufficiency, error, and sample ordering.

`breadth_tasks.py` validates the returned report directly. The
`_breadth_cache_diagnostics()` reconstruction helper is deleted.

## Module decomposition

### Production

- `services/derived_data_execution_policy.py`
  - mode enum, immutable policy, legacy-compatible resolver.
- `services/group_rank_models.py`
  - group prefetch data, immutable statistics, calculation result.
- `services/group_rank_input_loader.py`
  - group input-prefetch algorithm.
- `services/breadth_coverage.py`
  - breadth accumulator and immutable report.
- `tasks/group_rank_backfill_tasks.py`
  - manual range backfill, gap-fill, and one-year backfill Celery tasks.

`group_rank_tasks.py` imports and re-exports the three moved task objects.
Their registered Celery names remain:

- `app.tasks.group_rank_tasks.backfill_group_rankings`
- `app.tasks.group_rank_tasks.gapfill_group_rankings`
- `app.tasks.group_rank_tasks.backfill_group_rankings_1year`

Celery include configuration, API imports, beat schedules, and already queued
task names therefore remain compatible. Importing `group_rank_tasks` registers
the extracted tasks.

### Tests

- Add `test_derived_data_execution_policy.py` for the policy matrix, invalid
  modes, and legacy precedence.
- Add `test_group_rank_input_loader.py` for benchmark and stock provider
  exclusion plus typed prefetch statistics.
- Add `test_group_rank_backfill_tasks.py` for the three extracted manual tasks.
- Move guarded orchestration cases into
  `test_group_rank_execution_policy.py`.
- Keep ranking-algorithm and persistence tests in
  `test_group_rank_service.py`.
- Keep breadth formula tests in `test_breadth_calculator_service.py` and place
  accumulator/report contract tests in `test_breadth_coverage.py`.

The split is behavioral, not merely mechanical: each test module maps to one
production responsibility.

## Error and result behavior

- `refresh_guarded` breadth succeeds with usable stocks despite individual
  cache misses or insufficient histories; zero usable stocks or calculation
  errors remain failures.
- `refresh_guarded` group ranking succeeds with rankable groups despite missing
  constituents; missing benchmark or zero rankable groups remains a failure.
- `strict_cache_only` retains the existing breadth miss tolerance and group
  cache requirement.
- `auto` retains same-day cache-only warmup validation and historical provider
  fallback.
- Task result dictionaries retain `cache_only` and `cache_policy` fields where
  currently exposed.
- Moved manual task results and registered names do not change.

## Verification

Test-first implementation must cover:

1. Every execution-mode/date combination.
2. Both legacy flags, including strict precedence when both are true.
3. New pipeline signatures using `execution_policy="refresh_guarded"`.
4. Old Celery calls using legacy arguments.
5. Nested wrapper propagation using only the normalized mode.
6. Zero provider calls for guarded breadth target and gap-fill paths.
7. Zero provider calls for guarded group target and gap-fill paths.
8. Immutable group statistics and absence of caller-owned output mutation.
9. Deterministic breadth samples across different batch orders.
10. Identical breadth coverage semantics between daily and backfill paths.
11. Registration and import compatibility for extracted group backfill tasks.
12. Existing strict/manual/static/bootstrap behavior.

After focused tests pass, run the relevant backend unit suite, lint or compile
checks used by the repository, and the mandatory Beads/git completion workflow.

## Risks and controls

- **Queued-task compatibility:** retain legacy keyword arguments and test calls
  through each registered Celery task.
- **Task registration regression:** retain exact task names, import extracted
  tasks from the included module, and assert registration.
- **Result-shape regression:** serialize typed reports into existing external
  dictionary keys at task boundaries.
- **Policy drift:** permit legacy boolean names only in public task signatures
  and resolver calls; add a source-level regression assertion that downstream
  orchestration and services do not branch on them.
- **Large refactor masking behavior changes:** use red-green steps for policy,
  result contracts, loaders, and module moves; run provider-exclusion tests
  after each extraction.

## Acceptance criteria

- Legacy Celery calls remain operational and behaviorally unchanged.
- New internal orchestration uses one execution mode instead of boolean
  combinations.
- Group calculation has no mutable diagnostics output parameter.
- Breadth diagnostics have one service-owned source of truth with deterministic
  sampling.
- Group task and service files shrink through cohesive extraction, and the
  corresponding tests are split by responsibility.
- All issue-301 provider-exclusion, partial-coverage, strict-mode, manual-mode,
  bootstrap, and pipeline tests pass.
