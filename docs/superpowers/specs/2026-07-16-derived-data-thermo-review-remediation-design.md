# Derived-data thermo-review remediation

**Date:** 2026-07-16
**Status:** Approved for implementation
**Issue:** Beads `stockscreenclaude-2c1`
**Parent:** GitHub #301 / Beads `stockscreenclaude-duw`
**Builds on:** `2026-07-16-derived-data-execution-policy-refactor-design.md`

## Context

Issue #301 now correctly prevents refresh-guarded breadth and group-ranking
work from reaching price providers after the price-refresh guard. The first
maintainability refactor also introduced typed results, centralized coverage
diagnostics, and smaller task modules.

A thermo-nuclear review found four remaining structural problems:

1. `DerivedDataExecutionPolicy` stores a mode plus four freely constructible
   booleans, allowing contradictory states and leaving callers to interpret
   policy internals.
2. `GroupRankInputLoader` still locates dependencies at runtime and calls back
   into `IBDGroupRankService` for market caps, so the extraction has no clean
   ownership boundary.
3. The supposedly typed group-prefetch contract silently accepts legacy
   mappings and tuples inside core service/model code.
4. Breadth backfill copies invariant candidate and cache-miss symbol sets into
   an accumulator for every calculation date.

The review also identified that `IBDGroupRankService` remains a 1,806-line
facade containing calculation, persistence, queries, history, backfill,
dependency lookup, and compatibility logic.

This remediation keeps all approved issue-301 behavior while completing the
intended boundaries.

## Goals

- Make invalid execution-policy states unrepresentable.
- Make policy consumers ask the policy for decisions instead of branching on
  its fields.
- Give group input loading explicit, injected data sources with one owner per
  query.
- Preserve legacy group-prefetch compatibility only through a named adapter.
- Keep core group-prefetch models strictly typed.
- Separate invariant breadth price coverage from date-specific calculation
  outcomes.
- Move calculation and historical mutation responsibilities out of
  `IBDGroupRankService` while retaining its public API as a compatibility
  facade.
- Preserve all external behavior approved for issue #301.

## Non-goals

- Changing the price-refresh guard or its coverage threshold.
- Changing breadth formulas, RS formulas, ranking thresholds, or persistence
  schemas.
- Removing or renaming public Celery tasks or their existing keyword
  arguments.
- Removing public methods from `IBDGroupRankService`.
- Changing manual historical provider fallback.
- Introducing a general workflow framework, dependency-injection framework, or
  strategy-class hierarchy.
- Solving the separate orphaned/stale feature-snapshot lifecycle issue.

## Compatibility boundary

Compatibility is required at stable integration boundaries:

- existing Celery registered task names;
- existing Celery keyword arguments, including legacy cache-only flags;
- existing task result dictionary keys;
- existing public `IBDGroupRankService` method names and return shapes used by
  API, MCP, static export, snapshot, and task callers;
- existing static-export and manual backfill behavior.

Compatibility is not required for accidental internal test seams:

- monkeypatching `_prefetch_all_data()` to return an unnamed five-tuple;
- passing raw dictionaries into `GroupRankPrefetchData.stats`;
- patching a service-private market-cap query solely to affect the loader;
- directly constructing contradictory `DerivedDataExecutionPolicy` values.

Tests that use those seams will migrate to typed fixtures or an explicit
legacy adapter.

## Execution-policy design

### Closed state

`DerivedDataExecutionPolicy` stores only:

- `mode: DerivedDataExecutionMode`;
- `target_kind: DerivedDataTargetKind`, with `SAME_DAY` and `HISTORICAL`;
- `same_day_warmup_bypassed: bool`.

The constructor validates the only meaningful constraint: warmup bypass is
relevant only to same-day `AUTO` requests. Callers normally receive policies
from the resolver or named constructors rather than assembling values.

All operational decisions are derived properties:

- `allows_provider_reads`;
- `cache_only`;
- `requires_strict_completeness`;
- `requires_warmup_metadata`;
- `tolerates_partial_coverage`;
- `response_cache_policy`, returning `"refresh_guarded"` or `None`.

The mode remains available for serialization, but downstream code does not
combine it with independent flags.

### Validation profile

The policy exposes one validation profile:

- `PROVIDER_ALLOWED`;
- `STRICT_WITH_WARMUP`;
- `STRICT_WITHOUT_WARMUP`;
- `TOLERANT_CACHE_ONLY`.

Breadth and group task code selects its validation using this one value. It no
longer nests checks across `cache_only`, strictness, partial tolerance, and
warmup metadata.

### Gap-fill derivation

`policy.for_gap_fill()` owns the one deliberate difference between target-date
and automatic historical work:

- `AUTO` becomes provider-allowed historical behavior;
- `STRICT_CACHE_ONLY` remains strict cache-only;
- `REFRESH_GUARDED` remains tolerant cache-only.

Breadth and group wrappers pass `policy.for_gap_fill()` to their gap-fill
services. They do not construct a second policy or special-case `AUTO`.

### Response metadata

Policy-owned serialization helpers apply the existing external metadata:

- successful guarded responses receive `cache_only: true` and
  `cache_policy: "refresh_guarded"`;
- guarded error responses receive the same keys;
- non-guarded responses retain their current fields.

This removes repeated mode checks and the breadth task's
`"policy" in locals()` branch. Wrappers resolve the policy before entering
work that can fail, so error construction always has a request context.

### Legacy request flags

`force_cache_only` and `refresh_guarded_cache_only` remain accepted only by
public task entry points and the resolver. Existing precedence remains:

1. `force_cache_only=True`;
2. `refresh_guarded_cache_only=True`;
3. serialized `execution_policy`;
4. `AUTO`.

No service or orchestration branch reads the legacy flag names.

## Group input ownership

### Explicit source ports

`GroupRankInputLoader` receives five collaborators:

- `GroupRankUniverseSource.active_symbols(db, market)`;
- `GroupRankTaxonomySource.groups(db, market)` and
  `symbols_for_group(db, group, market)`;
- `GroupRankMarketCapSource.market_caps(db, symbols)`;
- the existing price and benchmark cache services.

Protocols and default SQL/application implementations live in
`group_rank_input_sources.py`:

- `StockUniverseGroupRankSource` adapts the existing stock-universe service;
- `IBDIndustryTaxonomySource` adapts `IBDIndustryService`;
- `SqlGroupRankMarketCapSource` performs the canonical market-cap query.

The composition root constructs these sources. The loader contains no import
from `wiring.bootstrap`, no direct static industry calls, and no fallback
market-cap implementation.

### Benchmark candidate contract

Cached benchmark fallback is part of the benchmark-cache interface used by the
loader. The loader calls `get_benchmark_candidates(market)` directly rather
than probing it with `getattr`. An empty candidate list falls back to the
primary benchmark symbol. Exceptions propagate through existing task retry or
error handling instead of being silently converted into a different benchmark
decision.

### Single ownership

The loader is the only owner of bulk ranking input assembly:

- normalize market;
- resolve benchmark;
- load active symbols;
- load group taxonomy;
- exclude inactive and unsupported symbols;
- load stock prices according to policy;
- load market caps;
- calculate prefetch statistics.

`IBDGroupRankService` does not duplicate those queries and does not provide
callbacks into the loader.

## Typed compatibility adapter

### Strict core model

`GroupRankPrefetchData` requires:

- `GroupRankPrefetchStats`;
- `frozenset[str]` active symbols;
- tuple-valued `symbols_by_group`;
- mapping-valued price and market-cap collections.

Its `__post_init__` validates these types and raises `TypeError` for legacy
containers. It does not rewrite a frozen instance with `object.__setattr__`.

`GroupRankPrefetchStats.from_mapping()` is removed from the core model.

### Explicit legacy adapter

`LegacyGroupRankPrefetchAdapter` owns old conversion rules:

- legacy five-tuples to `GroupRankPrefetchData`;
- legacy statistics mappings to `GroupRankPrefetchStats`;
- `spy_cached` to `benchmark_available`;
- missing legacy fields to their established defaults;
- list/set containers to tuples/frozensets.

Compatibility-facing facade methods may call the adapter where a legacy seam
must remain. Calculation, loader, and model code accept only typed inputs.

The adapter is independently tested and clearly marked for later removal. This
keeps compatibility visible without making every typed object dynamically
permissive.

## Group service decomposition

### Ranking calculator

Create `GroupRankingCalculator`, responsible for:

- vectorized RS calculation from prefetched prices;
- group metric aggregation;
- ranking and rank assignment;
- returning typed calculation output without database writes.

It receives the existing `RelativeStrengthCalculator` and contains the current
pure calculation helpers.

### Ranking repository

Create `GroupRankingRepository`, responsible for:

- upserting ranking rows;
- deleting ranking rows for a date range;
- current ranking queries;
- historical rank batch queries;
- group history and rank-mover queries where they operate directly on ranking
  persistence.

The repository uses the supplied SQLAlchemy session per call and retains the
existing query semantics and output dictionaries.

### Historical calculator

Create `GroupRankHistoricalCalculator`, responsible for:

- optimized range backfill;
- gap detection and optimized gap fill;
- chunked and legacy backfill entry points;
- trading-date generation;
- reusing one typed prefetch across date chunks;
- returning existing workflow statistics.

It composes `GroupRankInputLoader`, `GroupRankingCalculator`,
`GroupRankingRepository`, and the market calendar. The existing
`GroupRankHistoryBackfillService` remains the outer static-export workflow and
continues to call the compatibility facade.

### Compatibility facade

`IBDGroupRankService` remains the object returned by
`get_group_rank_service()`. Its public methods delegate:

- calculation to the input loader, calculator, and repository;
- read/history methods to the repository or the existing constituent-detail
  source;
- backfill/gap methods to the historical calculator.

Existing public signatures and result shapes remain unchanged. Private
calculation methods needed by no production caller are removed after tests move
to the focused components.

This decomposition targets responsibilities touched by issue #301. It does not
create a generic service layer or rewrite unrelated group-detail payload logic.

## Breadth coverage design

### Shared price coverage

Create immutable `BreadthPriceCoverage` containing:

- candidate stock count;
- symbols with cached history;
- cache-miss count;
- deterministic missing-symbol sample;
- cache coverage ratio.

`BreadthPriceCoverageAccumulator` owns only symbol sets and is instantiated once
per daily or backfill price-load operation.

### Per-date calculation outcomes

Create `BreadthOutcomeCounter` containing only integer counters:

- scanned observations;
- cache-miss observations;
- insufficient-history observations;
- error observations.

Backfill holds one small counter per requested date. It does not copy candidate,
cached, or missing-symbol sets per date.

### Composed report

`BreadthCoverageReport` composes one `BreadthPriceCoverage` and one
`BreadthOutcomeReport`. Existing convenience properties and serializers retain
the current external field names:

- `candidate_stocks`;
- `symbols_with_cached_history`;
- `cache_miss_stocks`;
- `cache_miss_symbols_sample`;
- `cache_coverage_ratio`;
- `total_stocks_scanned`;
- `skipped_stocks`;
- `insufficient_data_stocks`;
- `error_stocks`.

For daily calculation, the shared price coverage and the single daily outcome
form the report.

For historical backfill:

- each persisted date uses shared price coverage plus that date's outcome;
- the returned cache diagnostics use shared price coverage plus an aggregate
  outcome report;
- `insufficient_history_observations` remains the aggregate symbol-date count
  already exposed by the backfill result.

This preserves the distinction between unique-symbol cache coverage and
symbol-date calculation outcomes.

## Data flow

### Breadth wrapper

1. Resolve one policy at the public task boundary.
2. Derive `gap_policy = policy.for_gap_fill()`.
3. Load each symbol batch once using the gap or target policy.
4. Record symbol identity once in shared price coverage.
5. Record per-date outcomes in lightweight counters.
6. Compose coverage reports for validation, persistence, and diagnostics.
7. Apply policy-owned response metadata.

### Group wrapper

1. Resolve one policy at the public task boundary.
2. Derive `gap_policy = policy.for_gap_fill()`.
3. Loader obtains all typed inputs through injected sources.
4. Calculator produces rankings without persistence.
5. Repository persists or queries ranking rows.
6. Historical calculator reuses typed prefetch data across date chunks.
7. Facade and task boundary serialize existing external result shapes.
8. Apply policy-owned response metadata.

## Error handling

- Invalid serialized policy values retain the current task error behavior.
- Unsupported legacy prefetch shapes fail in
  `LegacyGroupRankPrefetchAdapter` with a descriptive `TypeError`.
- Core typed models reject untyped values immediately.
- Missing cached benchmark remains a meaningful group-ranking failure.
- Refresh-guarded breadth still fails only for calculation errors or zero usable
  stocks, not individual unavailable symbols.
- Refresh-guarded group ranking still fails for missing benchmark or zero
  rankable groups, not individual missing constituents.
- Provider-capable manual historical work retains existing provider retry
  behavior.
- Database and infrastructure failures continue through existing rollback,
  retry, and activity-reporting paths.

## Testing strategy

Implementation follows red-green-refactor cycles in this order:

1. Policy tests prove invalid combinations cannot be constructed, gap-fill
   derivation is centralized, and response metadata is policy-owned.
2. Task tests prove breadth and group wrappers no longer branch on multiple
   policy fields and preserve all public arguments/result fields.
3. Loader tests use fake source ports and prove there are no wiring imports,
   static taxonomy calls, duplicate market-cap queries, or service callbacks.
4. Adapter tests prove legacy tuples/mappings remain convertible while strict
   models reject them.
5. Calculator and repository characterization tests preserve group ranking,
   persistence, and query behavior while methods move.
6. Historical calculator tests preserve backfill/gap-fill results and provider
   exclusion.
7. Breadth tests prove one shared symbol coverage object is reused across dates,
   per-date reports remain correct, and aggregate backfill units are explicit.
8. Existing issue-301 provider-exclusion, partial-coverage, strict/manual,
   static-export, task-registration, and pipeline tests run unchanged or with
   typed fixture migrations only.

Source-level architecture tests assert:

- `DerivedDataExecutionPolicy` has no stored capability booleans;
- task modules do not special-case `AUTO` for gap-fill;
- group loader does not import from `wiring.bootstrap`;
- group loader does not reference `IBDGroupRankService`;
- core group models contain no legacy mapping coercion;
- breadth backfill does not create symbol-set accumulators per date.

## Acceptance criteria

- All `DerivedDataExecutionPolicy` instances are valid by construction.
- Consumers use policy methods/profiles rather than combinations of stored
  booleans.
- Breadth and group wrappers obtain gap behavior through
  `policy.for_gap_fill()`.
- `GroupRankInputLoader` has explicit injected sources and no callback into the
  facade.
- Legacy tuple/mapping support exists only in
  `LegacyGroupRankPrefetchAdapter`.
- Core prefetch models are strictly typed.
- Historical breadth tracks symbol sets once per load, not once per date.
- `IBDGroupRankService` is a delegating compatibility facade rather than the
  owner of calculation, persistence, and historical orchestration.
- Public Celery and service APIs, task names, result fields, and approved
  issue-301 behaviors remain compatible.
- Focused and backend quality gates pass.

## Risks and controls

- **Facade delegation changes call order:** characterization tests capture
  database writes, ranking results, and backfill statistics before methods move.
- **Legacy tests conceal production dependencies:** only production call sites
  determine compatibility requirements; test-only tuple seams migrate through
  the adapter.
- **Repository extraction changes transaction ownership:** repositories never
  commit independently. Existing task/service transaction boundaries remain
  authoritative.
- **Coverage unit confusion:** type and field names explicitly distinguish
  unique symbols from symbol-date observations.
- **Task response drift:** golden dictionary assertions cover guarded,
  strict-cache-only, same-day auto, historical auto, success, and error paths.
- **Oversized refactor:** changes land in independently testable commits for
  policy, breadth coverage, group input/adapter, calculator/repository,
  historical calculator, and facade cleanup.
