# Backend Test Baseline Remediation Design

## Context

The pull request's focused breadth and group-ranking tests pass, while the
comprehensive backend unit sweep reports 54 failures outside those feature
contracts. The failures cluster around test drift after intentional production
changes: market-catalog growth, key-market refresh targets, provider progress
callbacks, server authentication, synchronous FastAPI endpoints, OHLC
normalization, and sanctioned LLM fallback routing. One policy test falsely
flags the pull request's multiline `pct_change(..., fill_method=None)` call.

The default backend pytest command is also unsafe because legacy live-service
scripts perform HTTP, Redis, and database work at module import time. Current
GitHub Actions run only curated setup-engine gates, so neither the comprehensive
unit suite nor safe default collection is continuously enforced.

## Goals

- Restore the comprehensive hermetic backend test baseline to green.
- Preserve breadth/group cache-only behavior and existing compatibility paths.
- Update tests to assert canonical current contracts instead of historical
  literals or removed implementation details.
- Remove only assertions whose behavior was deliberately retired and is covered
  by the replacement contract.
- Keep live-service performance checks available through explicit opt-in
  execution without side effects during ordinary collection.
- Enforce the documented domain/use-case import boundaries.
- Add CI coverage that prevents the same drift from accumulating again.

## Non-goals

- Revert market exposure, the expanded market catalog, key-market refresh
  instruments, server authentication, synchronous database endpoints, OHLC
  validation, or LiteLLM provider routing.
- Change group-ranking or breadth completeness semantics.
- Run destructive live-service performance checks in pull-request CI.
- Add new application features.

## Approaches Considered

### 1. Restore old production behavior to satisfy tests

This would reduce immediate test edits but would remove intentional features and
weaken validation. It is rejected because most failures are stale tests, not
regressions.

### 2. Repair contracts by root-cause cluster and add a hermetic CI baseline

This is the selected approach. Tests will consume canonical registries and
planner outputs, mocks will implement current callable signatures, endpoint
fixtures will honor auth and sync/async boundaries, and data fixtures will meet
current normalization rules. Obsolete private-seam tests will be replaced by
public-contract coverage or removed when newer tests already cover the intended
behavior.

### 3. Delete failing suites and retain the curated CI gates

This is rejected because it would hide regressions in cache, auth, theme,
provider, and architecture behavior. The curated gates are useful but too narrow
to be the only backend signal.

## Design

### Canonical contract assertions

Market/bootstrap tests will derive expectations from the current market catalog,
bootstrap plan, task registry, and key-market instrument registry. Exact ordering
will remain asserted where ordering is a user-visible or orchestration contract.
Golden snapshots will be regenerated only after verifying the current catalog
order.

Refresh tests will distinguish active-universe symbols from appended Daily
Snapshot instruments. Progress totals, retry groups, and per-market accounting
will assert the complete planned workload instead of hard-coded universe counts.

### Current test seams

Provider test doubles will accept `progress_callback` and preserve existing
market arguments. Price-cache fixtures will use complete finite OHLCV frames.
Symbol endpoint fixtures will explicitly disable server authentication for tests
whose concern is validation. Direct calls to synchronous FastAPI endpoints will
be synchronous.

The line-oriented `pct_change` policy check will be replaced by an AST-based
call inspection that understands multiline keyword arguments and continues to
reject calls that omit `fill_method`.

### Retired behavior

The two legacy unbound-scan assertions that require a missing `FeatureRun` will
be removed because the supported behavior is fallback to `scan_results`, already
covered by newer tests. The direct private Gemini fallback test will be replaced
or removed because provider fallback is now owned by `LLMService`. The unused
in-process smart-refresh bypass helper and its test will be removed if repository
search confirms no caller; if a caller remains, progress publication will be
made safe without a Celery task id instead.

### Layer boundaries

The architecture test remains authoritative. Third-party calendar access and
bootstrap coverage evaluation will enter the snapshot use case through injected
domain-facing callables/ports. `UniverseType` will live in the domain and be
re-exported by the schema module for compatibility. Pure string-list
normalization will move to a domain utility with the infra module retaining a
compatibility re-export.

### Safe test collection

Legacy production-performance and cache-warming modules will not execute work at
import. Live checks will be normal pytest functions marked with a dedicated
`live_service` marker and skipped unless an explicit environment switch is set.
Collection must perform no network, Redis, or database mutation.

### CI

GitHub Actions will keep the existing fast curated gates and add a hermetic
backend unit-suite job. Explicit live, load, and chaos tests remain excluded.
The comprehensive job will use the same Python and dependency setup as the
current backend job. A collection-only check will guard against import-time exits
and side effects.

## Verification

Verification proceeds from the smallest affected clusters to the complete
hermetic suite:

1. Run each repaired test file while iterating.
2. Run the focused breadth/group suite to detect compatibility regressions.
3. Run `pytest tests/unit` and require zero failures.
4. Run safe default collection and the hermetic non-live suite.
5. Run existing Makefile backend gates and frontend checks affected by workflow
   changes.
6. Inspect the GitHub Actions workflow syntax and final git diff before push.

## Success Criteria

- The 54 reproduced unit failures are resolved without weakening current
  breadth/group behavior.
- Default pytest collection has no live-service side effects.
- Architecture boundary tests pass without allow-listing the violations.
- Existing focused and curated gates remain green.
- CI includes a comprehensive hermetic backend test signal.
