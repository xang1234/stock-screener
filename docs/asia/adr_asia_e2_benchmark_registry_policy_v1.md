# ADR ASIA-E2: BenchmarkRegistry Policy (v1)

- Date: 2026-04-11
- Status: Accepted (Target Architecture)
- Issue: `StockScreenClaude-asia.1.1`

## Context

SPY hardcoding makes non-US RS and benchmark-relative analytics invalid. Benchmarks must be market-aware and governed by explicit mapping rules.

## Decision

### Benchmark Mapping Policy

BenchmarkRegistry defines canonical benchmark mapping by market:

- US: SPY (default)
- HK: Hang Seng index-primary, ETF fallback as configured
- JP: Nikkei/TOPIX index-primary, ETF fallback as configured
- TW: TAIEX index-primary, ETF fallback as configured

Exact tickers are configuration-backed and versioned; selection policy is stable even if instruments change.

### Selection Semantics

- Benchmark resolution is performed by market via SecurityMaster output.
- Index-primary path is preferred for reference semantics.
- Fallback path (ETF or configured substitute) is explicit and observable.
- No non-US path may silently fall back to SPY.

### Operational Contract

- Health checks validate benchmark freshness per market.
- Cache keys/locks are benchmark-by-market scoped.
- Benchmark substitutions are auditable through config/version metadata.

## Consequences

- RS and benchmark-relative outputs become market-correct.
- Failures become isolated and diagnosable by market.
- Rollout gates can assert benchmark correctness objectively.

## Implementation Status (2026-04-11)

- Current runtime remains partially US-anchored in legacy benchmark cache paths.
- This ADR is binding for implementation tasks under `StockScreenClaude-asia.3.2`, `StockScreenClaude-asia.3.4`, and `StockScreenClaude-asia.6.*`.
- Required completion criteria before non-US launch:
  - market-scoped cache keys/locks and freshness checks,
  - benchmark resolution by market through SecurityMaster/BenchmarkRegistry,
  - explicit prohibition of implicit SPY fallback in non-US flows.

## Rejected Alternatives

- "Single global benchmark": rejected due to systematic bias and incorrect non-US semantics.
- "Hardcode per-service benchmark lists": rejected due to duplication and drift.
