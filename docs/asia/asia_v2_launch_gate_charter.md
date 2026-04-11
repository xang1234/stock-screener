# ASIA v2 Objective Launch-Gate Charter

- Date: 2026-04-11
- Status: Approved charter artifact for `StockScreenClaude-asia.1.3`
- Applies to: staged canary rollout under `StockScreenClaude-asia.11.*`
- Inputs from: ADR pack (`ASIA-E0..E5`), rollback policy (`asia_v2_flag_matrix_and_rollback_runbook.md`)

## Purpose

Define deterministic go/no-go gates so ASIA rollout cannot proceed on subjective judgment.
Each gate has:

1. Metric definition
2. Measurement window
3. Pass threshold
4. Hard-fail condition
5. Required evidence artifact

## Gate Policy

- All gates are blocking unless explicitly marked informational.
- US parity and non-US correctness are both required (no trade-off approval).
- Failing any hard gate blocks progression to the next canary stage.

## Gate Set

### G1. Schema/Contract Readiness (Hard)

- Metric: migration upgrade/downgrade success on production-like snapshot.
- Window: latest rehearsal run before canary.
- Pass: upgrade + downgrade complete without data-loss events.
- Hard fail: migration error, lock timeout breach, or unrecoverable downgrade.
- Evidence: migration rehearsal report (E11.2) + applied revision IDs.

### G2. Universe Integrity and Freshness (Hard)

- Metric A: by-market universe snapshot age.
- Metric B: deactivation safety thresholds and quarantine outcomes.
- Window: last 2 market sessions for each enabled market.
- Pass:
  - snapshot age <= 1 expected market session
  - no unquarantined destructive apply when thresholds are breached
- Hard fail: stale market snapshot > 1 session or destructive apply outside policy.
- Evidence: reconciliation diffs + quarantine/alert logs.

### G3. Benchmark/Calendar Correctness (Hard)

- Metric A: benchmark resolution by market (no SPY leakage for non-US).
- Metric B: expected trading date/session correctness by market.
- Window: regression suite + latest daily run.
- Pass:
  - 100% benchmark mapping assertions pass per market
  - calendar tests pass for US/HK/JP/TW session expectations
- Hard fail: any benchmark mismatch or calendar correctness regression.
- Evidence: benchmark test report + calendar test report.

### G4. Fundamentals Data Quality and Transparency (Hard)

- Metric A: completeness/provenance fields present for required scan metrics.
- Metric B: unsupported-field behavior correctly surfaced (non-US).
- Window: latest integration test run + canary telemetry sample.
- Pass:
  - required provenance fields present in 100% sampled responses
  - unsupported/computed reason codes emitted where expected
- Hard fail: silent fallback/coercion or missing provenance on required fields.
- Evidence: API contract tests + sampled payload audit.

### G5. Multilingual Extraction Quality (Hard)

- Metric: zh/ja/zh-TW golden-set precision/recall for theme+ticker extraction.
- Window: latest multilingual QA harness run.
- Pass:
  - precision >= 0.85
  - recall >= 0.75
  - false-positive rate <= 0.10
- Hard fail: any threshold miss.
- Evidence: QA harness summary artifact and confusion matrix.

### G6. US Parity and Non-US Scan Correctness (Hard)

- Metric A: US golden fixture parity against baseline.
- Metric B: non-US benchmark/percentile semantics regression checks.
- Window: latest pre-rollout regression run.
- Pass:
  - US parity suite fully green
  - non-US correctness suite fully green
- Hard fail: any parity/correctness regression.
- Evidence: consolidated test report from E6.5.

### G7. Performance and Stability (Hard)

- Metric A: scan create API p95 latency.
- Metric B: scan execution failure rate.
- Metric C: queue isolation under induced failure.
- Window: load/soak + fault-injection runs.
- Pass:
  - API p95 <= 1500 ms
  - scan execution failure rate <= 1.0%
  - isolated market fault does not breach unaffected market SLOs
- Hard fail: threshold breach on any metric.
- Evidence: load/soak report + fault-injection report.

### G8. Observability and Operations Readiness (Hard)

- Metric: required dashboards, alerts, runbooks, and owner mappings complete.
- Window: pre-canary checklist + runbook drill.
- Pass:
  - all required alerts wired and tested
  - runbook exercise completed with rollback drill
- Hard fail: missing alert coverage or untested runbook.
- Evidence: alert test logs + runbook drill record.

### G9. Rollback Control Validation (Hard)

- Metric: kill-switch matrix functional for each enabled market.
- Window: immediately before each canary step.
- Pass:
  - all subsystem kill-switch checks pass
  - emergency destructive-write stop validated
- Hard fail: any kill switch not working as documented.
- Evidence: flag-toggle test checklist results.

## Canary Progression Rules

1. HK canary may start only when G1-G9 pass.
2. JP canary may start only after HK canary passes and no unresolved SEV-1/SEV-2 incidents remain.
3. TW canary may start only after JP canary passes under same constraints.
4. GA requires all canaries green plus final signed gate artifact.

## Evidence Bundle (Required Attachments)

- Gate runner output (single signed pass/fail artifact)
- Migration rehearsal report
- Regression/parity reports (US + non-US)
- Multilingual QA report
- Performance/load/chaos reports
- Alert/runbook drill report
- Flag-toggle validation report

## Fail/Override Policy

- No unilateral override for hard gates.
- Temporary exception requires documented risk acceptance signed by platform + product owners and must include rollback deadline.
- Any exception expires before next canary stage.

## Ownership

- Gate owner: ASIA program owner (`StockScreenClaude-asia`)
- Technical approvers: Data Platform, Scanning, Themes, API/Frontend, SRE/Ops
- Final signoff recorded in `StockScreenClaude-asia.11.6`
