# Options Command Center thermo-review remediation

**Date:** 2026-09-04

**Status:** Approved direction; awaiting implementation

**Builds on:** `2026-09-04-options-command-center-design.md`

## Context

The Options Command Center implements the approved product behavior, including
the bounded Yahoo cohort, five-session continuity, live/static parity, and
last-good static fallback. A strict maintainability review found that the
implementation concentrates too many responsibilities in the refresh use case
and SQL repository, adds options-specific branches to the already-large static
site exporter, declares ports that do not describe the real dependencies, and
duplicates static publishing and wire-contract logic.

This remediation changes internal ownership without changing candidate
selection, metric formulas, publication thresholds, public API payloads,
database schema, task names, or static paths.

## Goals

- Keep the options run coordinator small and focused on workflow sequencing.
- Represent successful and unavailable candidate analysis as distinct types.
- Make dependency ports match the operations the application actually uses.
- Give transaction and run-state transitions one explicit owner.
- Separate command persistence, published queries, and history transfer.
- Remove options-specific orchestration from `StaticSiteExportService`.
- Reuse one atomic directory-publishing primitive for static artifacts.
- Make the backend schema the source of truth for frontend validation.
- Bring `StaticSiteExportService` back below 1,000 lines.

## Non-goals

- Changing Command Center behavior or presentation.
- Changing the 40-per-source, USD 100 million liquidity, continuity, or total
  cohort limits.
- Changing Yahoo request concurrency or the three-attempt symbol budget.
- Replacing SQLAlchemy, React Query, Pydantic, or the existing static-site
  architecture.
- Introducing a generic workflow or dependency-injection framework.
- Migrating existing database rows.

## Application decomposition

### Candidate cohort builder

`OptionsCandidateCohortBuilder` owns current and continuity input assembly. It
receives a typed candidate source, membership reader, and US session calendar,
then returns a typed cohort snapshot. The existing pure selection policy remains
unchanged.

### Candidate analyzer

`OptionsCandidateAnalyzer` owns expiration discovery, the three-attempt symbol
budget, provider normalization failures, metric calculation, history readiness,
quality evidence, assumptions, warnings, and bounded strike-point projection.
It returns exactly one of:

- `AvailableCandidateAnalysis`, containing the observation, metrics, evidence,
  assumptions, warnings, strike points, retry count, and readiness;
- `UnavailableCandidateAnalysis`, containing reasons, evidence, assumptions,
  warnings, and retry count.

The two types remove correlated nullable observation/metric fields and make
invalid result combinations unrepresentable. Retry ownership lives only here;
the Yahoo provider performs one provider operation per call.

### Run coordinator and transactions

`RefreshOptionsAnalyticsUseCase` owns only:

1. command validation and idempotent run acquisition;
2. staging the typed cohort;
3. bounded concurrent candidate analysis;
4. persisting completed analyses;
5. ranking and publication-policy evaluation;
6. one explicit terminal transition and retention request.

The repository exposes intent-level transaction methods rather than a public
`commit()`. Staging, assumptions, and terminal state changes are committed by
named repository operations. Cancellation transitions the run to `CANCELLED`
before returning and is checked while collecting completed candidates.

## Typed boundaries

The options ports are replaced with narrow protocols matching the real calls:

- `OptionsCandidateSource`;
- `OptionsMembershipReader`;
- `OptionsRunWriter`;
- `PublishedOptionsReader`;
- `OptionsHistoryGateway`;
- `OptionsProvider` and `SessionCalendar`.

Ports use domain/application DTOs, not SQLAlchemy models or `Any`. Unused Clock
and ProgressReporter ports are removed. Composition reuses the repository's
canonical cancellation token and a shared US session-window adapter.

Dividend provenance becomes an enum-backed value object so the yield and its
source cannot contradict one another.

## Persistence and history transfer

Persistence receives `PersistedCandidateAnalysis`, a typed projection created
at the application boundary. Metric and strike values are explicit fields
rather than dynamically named dictionaries.

The current SQL class is decomposed by responsibility:

- run command persistence;
- published read queries;
- history observation storage/transfer;
- retention.

These may share a session and private row mappers, but no public class combines
all four APIs. Strike points are replaced in one bulk operation per candidate,
avoiding per-strike lookup/update branching.

History transfer uses typed Pydantic models. The payload carries observations;
last-current memberships are derived from those observations on import and are
not duplicated in the payload. Checksums are calculated over the validated
model dump. Raw chains and strike points remain forbidden.

## Static-site integration

`StaticOptionsSection` owns live export, compatible fallback selection, stale
decoration, manifest contribution, and US metadata contribution. Both direct
static export and artifact combination call this component through the same
interface. `StaticSiteExportService` no longer constructs options dependencies
or mutates options manifest fields itself.

One `AtomicDirectoryPublisher` owns stage creation, backup, rename, rollback,
and cleanup. The market combiner, options exporter, options selector, and
fallback installer use it instead of maintaining parallel implementations.

Fallback download uses one artifact-candidate pipeline parameterized by artifact
name parsing, validation, and date extraction. Options and market artifacts do
not have separate download/install control flows.

## Frontend contract

Pydantic remains the authoritative wire model. The backend exports a committed
JSON Schema for the options manifest, command-center response, and symbol-detail
response. The frontend validates payloads against that generated schema through
one small adapter; handwritten field-by-field schema duplication is removed.
Query-key construction and safe static-path checks remain frontend concerns.

If introducing a schema-validator package would materially enlarge the runtime,
the fallback is a generated validator module checked into source. It must be
regenerable by one documented command and guarded by a drift test.

## Verification

Refactoring proceeds test-first, one boundary at a time. Existing behavioral
tests must remain unchanged except where they depended on accidental internal
seams. New tests cover:

- typed available/unavailable analysis results;
- persisted `CANCELLED` transition and cancellation during collection;
- exact port conformance at composition time;
- typed history round-trip and derived membership continuity;
- atomic publisher rollback;
- identical options static output through direct and combined export paths;
- generated-schema drift and live/static frontend contract parity;
- file-size guard keeping `StaticSiteExportService` below 1,000 lines.

Final verification includes focused backend tests after each extraction, the
complete backend unit suite, frontend tests and lint, production frontend build,
and `git diff --check`.
