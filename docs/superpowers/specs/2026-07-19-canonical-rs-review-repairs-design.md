# Canonical RS Review Repairs

**Date:** 2026-07-19

**Status:** Approved architecture — written spec awaiting review

**Issue:** `stockscreenclaude-stm`

**Parent design:**
[`2026-07-18-canonical-market-rs-design.md`](2026-07-18-canonical-market-rs-design.md)

## Context

The canonical Market RS implementation introduced the intended balanced,
per-horizon percentile formula and carried its metadata through Scan, Groups,
static export, and RRG. A thermo-nuclear quality review found five correctness
gaps and several maintainability blockers in the implementation:

1. Static refresh selects balanced stock RS, then invokes a legacy-only Group
   history backfill that stores `legacy-linear-v1` rows by default. The exporter
   subsequently asks for an exact balanced Group snapshot and can find none.
2. Feature-row Group enrichment does not select US rows by formula and
   independently recomputes non-US ranks from serialized feature rows. Scan
   metadata can therefore disagree with the canonical Group table.
3. Static artifact combination parses a requested RS formula but does not pass
   it to the combiner, and the workflow does not provide it at combine time.
   Current and fallback artifacts can be combined without the intended formula
   guard.
4. Custom Scan evaluates its RS filter only when benchmark data is present,
   even when the stock already has canonical RS. Canonical hydration is also a
   dynamic, duplicated convention rather than part of the data-provider port.
5. The Market RS input loader substitutes raw `close` when `adj_close` is
   absent. A split or distribution can therefore contaminate a percentile
   snapshot that is documented as adjusted-price based.

The review also identified large, multi-responsibility files around the new
workflow. Those structures make it too easy for another consumer to bypass the
formula/date identity or grow a parallel calculation path.

This document refines the parent design. The formula mathematics, public API
schemas, static schema v3, and legacy rollback behavior remain unchanged.

## Decision

Use targeted consolidation around one explicit snapshot identity and one
formula-aware Group coordinator. Every Group producer and reader will carry
`market + as_of_date + formula_version`; balanced snapshots additionally prove
their canonical Market RS run ID. Feature enrichment will become a pure reader
of that stored snapshot. Scanner hydration will become a typed provider
contract. Static combination and Market RS input eligibility will fail closed.

This is preferred over a narrow call-site patch because a local formula
argument would leave the competing Group calculation paths and dynamic scanner
hydration intact. It is preferred over a full ranking-system rewrite because
the existing database model, payload contracts, RRG mathematics, and legacy
engine are valid compatibility boundaries.

## Goals

1. Make balanced and legacy Group creation formula-aware in live, static, and
   historical workflows.
2. Make one stored Group snapshot the source of Group rank metadata for every
   Market.
3. Prevent static bundles from mixing current or fallback artifacts from a
   different formula.
4. Let canonical RS filtering work without benchmark bars while retaining the
   benchmark-dependent legacy path.
5. Require adjusted-close values at every Market RS session anchor.
6. Recover cleanly after a failed historical date and continue with later
   dates when the caller requests best-effort backfill.
7. Reduce the oversized orchestration files into cohesive, testable modules
   without changing public payloads.
8. Preserve `legacy-linear-v1` rollback and
   `balanced-horizon-percentile-v2` activation semantics.

## Non-goals

- Change the 20/30/20/15/15 horizon weights or percentile mathematics.
- Change Group rank ordering, constituent minimums, or RRG transformations.
- Add another RS formula version.
- Recalculate Group rankings from a scan, watchlist, index, or serialized
  static subset.
- Remove legacy rows or legacy rollback tooling.
- Change the static v3 JSON shape or the public live API shape.
- Refactor unrelated scanner, export, or frontend behavior.

## Required invariants

### Snapshot identity

Introduce an immutable `GroupSnapshotIdentity` value object with exactly:

```text
market
as_of_date
formula_version
```

Construction normalizes `market` to the catalog's uppercase code and rejects a
blank formula. A balanced Group snapshot is valid only when all of its rows
also reference the one completed `MarketRsRun` for that identity. The database
uniqueness constraint remains:

```text
(industry_group, date, market, rs_formula_version)
```

The Market RS run remains unique by:

```text
(market, as_of_date, formula_version)
```

The snapshot identity is required in application and service interfaces. Only
outer compatibility boundaries may translate omitted historical metadata to
an explicit `legacy-linear-v1` identity, and that inference must be returned as
a diagnostic. Balanced is never inferred from missing metadata.

### Formula isolation

- A balanced Group row is built only from
  `StockRsSnapshot` rows belonging to its exact completed Market RS run.
- A legacy Group row is built only by the existing legacy ranking engine.
- A query for one formula never treats a row from another formula as coverage.
- RRG receives history from one explicit formula only.
- Feature enrichment and static export do not calculate Group rank.
- Static combine accepts a market artifact only when its formula matches the
  requested formula.

### Price eligibility

Market RS uses the database `adj_close` field at current, 1M, 3M, 6M, 9M, and
12M session anchors. `MarketRsInputLoader` does not substitute `close` during
calculation. Upstream ingestion remains responsible for putting a valid,
provider-normalized adjusted value in `adj_close`.

A missing or invalid adjusted current anchor contributes to the current-price
coverage failure. A missing historical adjusted anchor excludes that symbol
from the common eligible universe. A benchmark candidate missing any adjusted
anchor is ineligible as the benchmark. Diagnostics distinguish these cases
with stable reason codes.

## Backend architecture

### 1. Group snapshot coordinator

Add `GroupRankSnapshotCoordinator` as the only application service that ensures
Group rows for an identity. It receives injected collaborators for canonical
Market RS creation, canonical Group aggregation, legacy Group calculation, and
exact snapshot reads.

Its public operations are:

```python
ensure_snapshot(db, *, identity) -> GroupSnapshotResult
backfill(db, *, identities, continue_on_error) -> GroupBackfillReport
```

`ensure_snapshot` behaves as follows:

- Load exact existing rows for the identity.
- If complete rows exist, validate their identity and return them without
  recalculation.
- For `balanced-horizon-percentile-v2`, ensure the exact completed Market RS
  run, call `CanonicalGroupRankingService`, then reload and validate the stored
  Group rows including `market_rs_run_id`.
- For `legacy-linear-v1`, delegate to the existing legacy engine while passing
  the formula explicitly, then reload exact legacy rows.
- Reject unsupported formulas rather than falling back.
- Treat an empty but successfully calculated snapshot as an explicit result
  with diagnostics, not as coverage by another formula.

`backfill` processes identities in ascending date order. Every date has an
exception boundary. On failure it calls `db.rollback()` before recording the
error or proceeding, so a poisoned SQLAlchemy session cannot make all later
dates fail. `continue_on_error=False` re-raises after rollback; the static
historical best-effort path uses `True` and reports every failed date. A date is
counted as processed only after an exact snapshot reload succeeds.

The coordinator does not make the Market RS run and Group snapshot one
transaction: Market RS is independently publishable and may remain available
when Group aggregation fails. Group writes for one date remain atomic.

### 2. Formula-aware history backfill

`GroupRankHistoryBackfillService.backfill` gains a required
`formula_version`. Its coverage query filters by Market, date, and formula. Its
gap-filler protocol receives `formula_version`, and its implementation delegates
to `GroupRankSnapshotCoordinator.backfill` instead of directly calling
`IBDGroupRankService.fill_gaps_optimized`.

`IBDGroupRankService.fill_gaps_optimized` remains available only as the legacy
implementation behind the coordinator. Its storage calls must receive
`LEGACY_RS_FORMULA_VERSION` explicitly; no newly touched call relies on a
default formula.

Live scheduled tasks, manual Group refresh, static daily refresh, rollout
backfill, and RRG history preparation all construct the same sequence of
`GroupSnapshotIdentity` values and use this service.

### 3. Stored Group snapshot reader

Extract an exact reader with these operations:

```python
load_exact(db, *, identity) -> list[GroupRankRow]
load_rank_map(db, *, identity) -> dict[str, int]
available_dates(db, *, market, formula_version, through_date) -> tuple[date, ...]
```

Every predicate includes `rs_formula_version`. For balanced rows the reader
validates that all non-empty rows share one non-null `market_rs_run_id` and that
the referenced completed run matches Market, date, and formula. Mixed run IDs
or formula metadata raise a typed integrity error.

This reader is used by live Group responses, feature enrichment, static Groups,
RRG history, and rollout validation. Payload conversion remains separate from
database selection.

### 4. Feature-row Group enrichment

`_enrich_feature_run_with_ibd_metadata` resolves the feature run first, then
requires its persisted `config_json.rs_formula_version`. It builds the exact
snapshot identity from the feature run's Market, supplied ranking date, and
that formula.

The Market-specific behavior is limited to taxonomy resolution:

- US continues to use the IBD symbol-to-group mapping.
- Other Markets continue to use `MarketTaxonomyService` for group, sector,
  industry, and themes.

All Markets obtain `ibd_group_rank` from `load_rank_map(identity)`. The non-US
call to `compute_group_rankings_from_serialized_rows` is removed. The US query
that omits formula is removed. If the exact Group snapshot is absent,
enrichment raises `GroupSnapshotUnavailable` before modifying any feature row.
Task callers report the enrichment as skipped or failed without erasing a
previous valid rank. A stock whose mapped Group is legitimately absent from an
otherwise valid snapshot receives a null rank and contributes to the existing
`missing_rank_rows` diagnostic.

New feature runs always persist explicit RS formula/date/run metadata before
enrichment. A `FeatureRunRsIdentityResolver` handles old rows deterministically:
when formula, RS run ID, and RS as-of metadata are all absent it returns an
explicit legacy identity plus `identity_source: inferred_legacy`; partial
canonical metadata without a formula is an integrity error. It cannot infer
balanced.

### 5. Static refresh order

For each Market, the static refresh order becomes:

```text
price refresh
  -> exact Market RS snapshot
  -> exact current and historical Group snapshots
  -> feature snapshot
  -> exact Group enrichment
  -> static Groups and formula-filtered RRG export
```

The Group history step receives the selected `rs_formula_version`. Balanced
static refresh therefore cannot pass through the legacy gap filler. The
post-build enrichment pass remains idempotent for already-published feature
runs, but it reads the same exact stored Group snapshot.

Static Group payload creation moves to a focused
`StaticGroupSectionBuilder`. It loads exact current and historical Group rows
through the stored snapshot reader, validates feature-run metadata, and builds
the existing v3 payload. It may use serialized stock rows for constituent
details and top-stock presentation, but never for Group rank calculation.

### 6. Static combine formula guard

`export_static_site.py --rs-formula-version` applies in both refresh/export and
`--combine-artifacts-dir` modes. In combine mode the CLI creates an expected
formula map for every configured static Market and passes it as
`rs_formula_version_overrides`.

The static workflow's combine job passes the same `RS_FORMULA_VERSION` used by
the per-Market build jobs. `StaticSiteExportService.combine_market_artifacts`
delegates artifact selection to a focused `StaticArtifactCombiner` with these
rules:

1. Prefer the current artifact only if its Market entry, manifest, Groups, and
   RRG metadata are compatible with the expected formula.
2. Consider the fallback artifact only when current is absent and fallback is
   compatible with the same formula.
3. Reject a present but incompatible artifact with a Market-specific error; do
   not silently omit it and do not fall back across formulas.
4. Abort the combined publish when any required static Market has no compatible
   current or fallback artifact.
5. Build the combined manifest only from the validated selections.

The existing no-current-artifact fallback for a Market holiday or failed daily
refresh remains supported, provided the fallback uses the requested formula.

### 7. Typed scanner RS hydration

Extend the `StockDataProvider` port with:

```python
apply_market_rs_resolution(
    results: dict[str, object],
    resolution: MarketRsResolution,
) -> None
```

`DataPrepStockDataProvider` implements it by delegating to
`DataPreparationLayer`. Test doubles implement the same contract. Move
`MarketRsResolution` before the port declaration or use a forward annotation so
the interface is statically valid.

`RunBulkScan`, `BuildDailySnapshot`, and `ScanOrchestrator` call the method
directly. Their `getattr`, `setattr`, and duplicated hydration fallbacks are
removed. A provider that cannot hydrate RS fails at construction or in its
contract test rather than silently taking a different runtime path.

`CustomScanner` resolves RS before checking benchmark availability:

- When canonical balanced ratings are attached, the RS filter uses them and
  does not require benchmark bars.
- When the active mode is legacy, its lazy calculator still requires stock and
  benchmark price series.
- When neither source is available, it returns the existing structured
  insufficient-data result instead of omitting the enabled filter.

`needs_benchmark=True` may remain in data requirements for legacy compatibility;
it is no longer a gate around canonical evaluation.

### 8. Strict adjusted-close inputs

`MarketRsInputLoader` queries and accepts only `StockPrice.adj_close`. It removes
the row-level `adj_close if present else close` fallback. The run diagnostics
record `price_basis: adj_close_only` so staged rollout validation can prove the
corrected input policy was used.

Stable diagnostics include:

- `benchmark_adjusted_anchor_missing` for a benchmark with one or more missing
  adjusted anchors;
- `current_adjusted_price_coverage_below_threshold` for the Market coverage
  gate; and
- per-symbol exclusions such as
  `missing_adjusted_current_session_anchor` and
  `missing_adjusted_63_session_anchor`.

Completed balanced runs whose diagnostics do not identify the
adjusted-close-only input policy are incompatible. `MarketRsReader` and the
Group coordinator reject them. `MarketRsSnapshotService.calculate` raises
`MarketRsSnapshotIncompatible` for such a run unless its caller explicitly sets
`rebuild_incompatible=True`. The explicit rollout/backfill and static build
paths set that option, atomically clear the run's stock rows, recalculate the
same unique Market/date/formula run, and publish it only after the adjusted-
close result passes normal completion checks. Ordinary read paths never mutate
or consume it. Rollout validation also rejects a staged run without the marker.
Legacy rows are untouched.

### 9. Typed RRG unavailability and activation integrity

`StaticGroupsRRGUnavailableError` gains a stable reason code. At minimum the
codes distinguish:

- RRG not enabled for a Market;
- insufficient or absent formula-filtered history;
- formula mismatch;
- date mismatch; and
- source/schema unavailable.

Rollout validation branches on the code instead of matching substrings in an
English error message. Insufficient history remains an allowed unavailable RRG
state; formula/date/source mismatches remain validation failures.

The activation validator returns the staged manifest hash. Immediately before
switching the active formula pointer, activation recomputes that hash and
refuses the switch if the directory changed, disappeared, or no longer validates.
This closes the validation-to-activation race without changing the pointer
schema.

## Structural decomposition

The repair extracts cohesive units while preserving existing import facades and
public call signatures where callers outside this feature depend on them.

### Backend

- `group_rank_tasks.py` retains Celery/task boundaries and delegates identity,
  range, and formula decisions to the Group snapshot coordinator.
- `bootstrap.py` moves Market RS and Group snapshot construction into a focused
  wiring module and exposes the same process-scoped accessors.
- `ibd_group_rank_service.py` becomes a compatibility facade over separate
  legacy calculation/gap-fill and exact history-read services. No canonical
  calculation is added to the legacy engine.
- `static_site_export_service.py` retains top-level export coordination while
  Group section construction and artifact combination move to focused services.
- `market_rs_rollout_service.py` becomes a small facade over backfill,
  validation, and activation components.

No production Python module modified by this repair may remain above 1,000
lines, and no new extracted module should exceed 700 lines. The line limit is a
guardrail; each extraction must also have one clear reason to change and direct
unit tests.

### Frontend

Extract the live/static Group table, column definitions, and Group detail UI
from `GroupRankingsPage.jsx`. Live and static modes share the same RS/1M RS/3M
RS field definitions and formatters while retaining their existing sort and
interaction differences. `GroupRankingsPage.jsx` becomes the data-loading and
mode orchestration page and remains below 1,000 lines.

## Error handling and observability

- Unsupported or mixed formulas raise typed errors containing Market, date,
  and requested formula.
- Historical reports contain one result per requested date with `processed`,
  `existing`, `empty`, or `errored` status.
- A caught database exception is always followed by `rollback()` before reuse
  of that session.
- Balanced Group diagnostics include Market RS run ID and universe size.
- Static combine errors identify whether current, fallback, or both artifacts
  were incompatible.
- No error policy depends on message substring matching.
- Logs may add context, but callers make decisions from result fields or reason
  codes.

## Compatibility

- Public Scan and Group API response fields are unchanged.
- Static schema remains v3; no database migration is required.
- Existing `avg_rs_rating_1m`, `avg_rs_rating_3m`, formula, and Market RS run
  fields remain authoritative.
- `legacy-linear-v1` remains selectable and its historical engine remains
  callable through the new coordinator.
- Balanced and legacy snapshots continue to coexist under existing unique
  constraints.
- RRG mathematics and payload coordinates do not change for identical input
  history.
- Existing service imports that are part of task or test boundaries are kept as
  forwarding facades during extraction.

## Test-first implementation

Each production change starts with a focused failing regression test.

### Group identity and backfill

- A balanced static history request invokes canonical Market RS and canonical
  Group aggregation and never invokes the legacy gap filler.
- A legacy request explicitly invokes the legacy path.
- Coverage for one formula does not satisfy another formula.
- Existing exact rows are reused only when balanced Market RS run IDs match.
- One failed date rolls the session back; a later date can still succeed.
- Empty and unsupported-formula results are explicit.

### Feature enrichment and parity

- US enrichment ignores same-date rows from a different formula.
- Non-US enrichment reads stored Group ranks rather than recomputing serialized
  rows.
- US, HK, TW, and KR enrichment use the feature run's exact formula identity.
- Missing formula metadata cannot become balanced implicitly.
- A golden fixture produces identical Group ranks in stored rows, Scan details,
  the live payload, and the static payload.

### Static refresh, combine, and RRG

- Static balanced refresh creates exact balanced Group history before feature
  enrichment.
- The combine CLI passes the requested formula map.
- The workflow combine command supplies `RS_FORMULA_VERSION`.
- A current artifact with the wrong formula fails.
- A fallback artifact with the wrong formula fails.
- A compatible fallback still works when a Market has no current artifact.
- Typed insufficient-history RRG is allowed; formula/date/source errors fail.
- Activation fails if staged content changes after validation.

### Scanner contract

- `StockDataProvider` contract tests require RS hydration.
- Bulk Scan, daily snapshot, and direct orchestrator paths hydrate identically.
- Canonical RS 87 with an empty benchmark passes an RS minimum of 80.
- Legacy RS with an empty benchmark returns insufficient data.
- An enabled RS filter can never disappear from the result silently.

### Adjusted-close policy

- Raw `close` cannot satisfy a missing adjusted current or historical anchor.
- A benchmark with raw close but missing adjusted close is rejected.
- Coverage and exclusion reason codes identify adjusted-price failures.
- Completed balanced runs carry `price_basis: adj_close_only` diagnostics.
- Read paths reject a completed balanced run lacking that marker without
  mutating it.
- Explicit backfill rebuilds an incompatible run and rollout validation rejects
  any staged run still lacking the marker.

### Structure and UI

- Extracted service tests cover their public behavior without task or Flask app
  setup.
- Shared Group table tests assert overall, 1M, and 3M columns in live and static
  modes.
- Existing live sorting, static rendering, Group details, and RRG tests remain
  green.
- A final line-count check confirms every modified production module is within
  the stated guardrail.

## Verification gates

The completed repair must pass:

```bash
cd backend && source venv/bin/activate && pytest
cd frontend && npm run test:run
cd frontend && npm run lint
git diff --check
```

Focused regression tests run first during development. Full backend and frontend
gates run before the issue is closed and the branch is pushed.

## Acceptance criteria

1. Static balanced refresh creates and exports an exact balanced Group snapshot
   for every publishable Market/date, including required history.
2. No balanced Group path invokes legacy storage defaults.
3. Feature rows in every Market use the rank from the exact stored Group
   snapshot matching the feature run's formula.
4. Scan, live Groups, static Groups, and RRG cannot mix Group rows across
   formulas.
5. Static combination fails when any selected current or fallback Market
   artifact uses a formula other than the requested one.
6. Canonical Custom Scan RS filtering works when benchmark bars are empty; the
   legacy path still requires its benchmark.
7. Every Stock data provider implements the typed Market RS hydration contract;
   no dynamic hydration fallback remains.
8. Market RS ignores raw close values and reports adjusted-price coverage and
   exclusion failures explicitly.
9. Backfill can process a later date after an earlier date raises a database or
   calculation error.
10. RRG validation uses typed reason codes, and activation refuses changed
    staged artifacts.
11. Live and static Group tables retain overall, 1M, and 3M RS columns and
    identical values for a fixed snapshot.
12. Public payload schemas, static v3, formula versions, legacy rollback, and
    RRG mathematics remain compatible.
13. The oversized files named in this design are decomposed within the stated
    line-count and responsibility guardrails.

## Delivery sequence

1. Add snapshot identity, reader, and coordinator tests and implementation.
2. Route history backfill, live tasks, and static refresh through the
   coordinator.
3. Replace feature-row Group recomputation with exact stored reads.
4. Wire static combine formula enforcement and typed RRG failures.
5. Make provider hydration typed and repair Custom Scan's canonical path.
6. Enforce adjusted-close-only inputs and rollout validation.
7. Complete backend/frontend extractions behind compatibility facades.
8. Run parity, integration, full quality, and rollout checks.
9. Update the parent design/README only where this repair changes operational
   wording, then close the Beads issue and push the verified branch.
