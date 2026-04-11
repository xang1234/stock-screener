# ADR ASIA-E5: ReconciliationCircuitBreaker Policy (v1)

- Date: 2026-04-11
- Status: Accepted
- Issue: `StockScreenClaude-asia.1.1`

## Context

Universe updates for new markets are vulnerable to source anomalies, parser regressions, and accidental mass removals. A deterministic safety policy is required before destructive changes are applied.

## Decision

### Reconciliation Workflow

For each market snapshot refresh:

1. Build canonical candidate snapshot
2. Diff against prior snapshot (add/remove/change)
3. Evaluate safety thresholds
4. Apply if safe, otherwise quarantine and alert

### Circuit-Breaker Conditions

Block destructive apply when threshold checks fail, including:

- minimum universe count violation
- maximum removed-percentage breach
- anomaly-score breach on unexpected churn

### Governance Contract

- Diff artifacts are persisted and reviewable.
- Quarantined snapshots are non-destructive and operator-visible.
- Alerts map to runbook actions and rollback controls.

## Consequences

- Provider anomalies are less likely to cause catastrophic universe churn.
- Launch and operations teams gain deterministic forensic artifacts.
- Rollout blast radius is reduced through explicit safety gates.

## Rejected Alternatives

- "Always apply latest source": rejected due to high destructive risk.
- "Manual-only reconciliation": rejected due to operational latency and inconsistent enforcement.
