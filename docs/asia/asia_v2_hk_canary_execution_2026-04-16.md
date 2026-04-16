# ASIA v2 HK Canary Execution Record — 2026-04-16

- Bead: `StockScreenClaude-asia.11.3`
- Playbook: `asia_v2_hk_canary_playbook.md`
- Verdict: **PASS** (9/9 hard gates)
- Mode: **dress-rehearsal** against ephemeral Postgres + synthetic external evidence
- Status: **verification rerun complete** — original bead closure remains valid

## Why this rerun exists

This rerun was executed after the JP and TW rehearsals, using the same Postgres-backed canary harness, to confirm the original HK dress rehearsal still passes under the current branch state. The HK bead remains closed; this record is an additional verification snapshot, not a replacement for the original 2026-04-15 execution record.

## Evidence mix

- **Real repo evidence** for gates whose evidence already lives in the repo: G1 (migration rehearsal), G3 (benchmark registry), G8 (runbook + drill), G9 (flag matrix)
- **Seeded Postgres telemetry** for the DB-backed gates: G2 (universe drift) and G4 (completeness) consumed deterministic rows inserted into a fresh PostgreSQL `market_telemetry_events` table
- **Labeled synthetic evidence** for the externally-produced gates: G5 (multilingual QA), G6 (parity regression), G7 (load/soak) reused from `data/governance/canary_evidence/hk-2026-04-15/`

## Environment

| Item | Value |
|---|---|
| Postgres | `postgres:16` ephemeral container `asia-canary-hk-pg` on port 55434 |
| Schema | head — all migrations applied via `alembic upgrade head` |
| Gate runtime | `backend/scripts/run_market_canary_rehearsal.py` invoking `backend/app/services/governance/launch_gates.py` |
| Seeded telemetry | 4 `universe_drift` rows (worst ratio = 0.008 on HK), 4 markets × 1 `completeness_distribution` row each (worst low-bucket = 0.04 on TW), plus HK `freshness_lag` and `benchmark_age` rows |
| Evidence dir | `data/governance/canary_evidence/hk-2026-04-15/` |
| Fixed runner clock | `2026-04-16T20:00:00+00:00` |

## Procedure executed

The physical flag-toggle sequence from the HK playbook was not replayed in a live environment because there is no real HK production target inside this repo sandbox. What was executed:

- **Stage 0 (preconditions)**: started a fresh `postgres:16` container and applied migrations to head
- **Stages 1–6 (post-rollout state)**: seeded HK-focused telemetry and attached the existing HK external evidence bundle
- **Stage 7 (post-canary verdict)**: ran the checked-in harness, rendered the signed artifact trio (`.json`, `.md`, `.sha256`), and verified the sidecar hash with `sha256sum -c`

## Per-gate results

| Gate | Name | Status | Detail |
|---|---|---|---|
| G1 | Schema/Contract Readiness | **pass** | Picks up `asia_v2_e11_st2_migration_rehearsal_report_2026-04-15.md` with no-data-loss assertion |
| G2 | Universe Integrity and Freshness | **pass** | Worst HK universe drift ratio in last 2d = **0.008** (< 0.15 critical) |
| G3 | Benchmark/Calendar Correctness | **pass** | Registry maps US→SPY, HK→^HSI, JP→^N225, TW→^TWII; HK benchmark remains anchored on `^HSI` |
| G4 | Fundamentals Data Quality | **pass** | Worst market completeness 0-25 bucket = **0.04** on TW (< 0.50 critical); HK low-bucket remains **0.02** |
| G5 | Multilingual Extraction Quality | **pass** | precision=0.910, recall=0.820, fpr=0.060 (thresholds 0.85/0.75/0.10) - synthetic |
| G6 | US Parity and Non-US Scan Correctness | **pass** | `us_parity_pass=true`, `non_us_correctness_pass=true` - synthetic |
| G7 | Performance and Stability | **pass** | p95=1180ms, failure_rate=0.004, market_isolation=true - synthetic |
| G8 | Observability and Operations Readiness | **pass** | Runbook present; drill `asia_v2_runbook_drill_2026-04-15.md` is **1 day** old (<= 14 day budget) |
| G9 | Rollback Control Validation | **pass** | Flag matrix references all 6 required kill switches |

## Signed artifact

Because the original HK artifact already exists at `data/governance/launch_gates/2026-04-15-pass.*`, and 2026-04-16 already has separate JP and TW audit records, this verification rerun writes its signed artifact under a market-scoped subdirectory:

```
data/governance/launch_gates/hk-2026-04-16/2026-04-16-pass.json
data/governance/launch_gates/hk-2026-04-16/2026-04-16-pass.md
data/governance/launch_gates/hk-2026-04-16/2026-04-16-pass.sha256
```

| Hash | Value |
|---|---|
| `content_hash` (semantic, inside JSON) | `6e647458c168867ead9b3ddef184144c85489c5208b71316a5e071e94afab721` |
| File hash (sidecar, sha256sum-compatible) | `26bd7f869c1cf9b72ca7c0c4200297d3e3d099c972d056c30da1b334ef9933cf` |

Verification was performed with `sha256sum -c 2026-04-16-pass.sha256`, which returned `OK`. A direct semantic-hash recompute against the JSON payload also matched.

## Residual risk

This rerun revalidates the DB-backed rehearsal path for HK under the current branch state. The remaining dress-rehearsal caveat is unchanged from the original HK run: G5/G6/G7 still use synthetic evidence files rather than production-generated outputs.

## Defects found during the rerun

| # | Description | Resolution |
|---|---|---|
| — | None | The current branch state still produces a clean HK PASS under the Postgres-backed rehearsal harness. |

## Decision

> **PASS** - The HK verification rerun completed against real Postgres and produced a signed PASS artifact.

This rerun is a check only. It does not reopen or alter the already-closed `asia.11.3` bead.
