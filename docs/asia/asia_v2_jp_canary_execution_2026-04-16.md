# ASIA v2 JP Canary Execution Record — 2026-04-16

- Bead: `StockScreenClaude-asia.11.4`
- Playbook: `asia_v2_hk_canary_playbook.md` (JP replay)
- Verdict: **PASS** (9/9 hard gates)
- Mode: **dress-rehearsal** against ephemeral Postgres + synthetic external evidence
- Status: **DB-backed rehearsal complete** — supersedes the earlier provisional synthetic-only run

## Why this record replaced the earlier provisional run

The first JP record from 2026-04-16 was explicitly provisional because Docker access was unavailable, so the DB-backed gates were exercised only through an in-process fake session. This rerun executed the checked-in canary harness against a fresh `postgres:16` container and replaced that provisional record with the real rehearsal path.

This now matches the HK rehearsal shape: real Postgres migrations, real inserts into `market_telemetry_events`, normal launch-gate evaluation, and signed artifact verification on disk.

## Evidence mix

- **Real repo evidence** for gates whose evidence already lives in the repo: G1 (migration rehearsal), G3 (benchmark registry), G8 (runbook + drill), G9 (flag matrix)
- **Seeded Postgres telemetry** for the DB-backed gates: G2 (universe drift) and G4 (completeness) consumed deterministic rows inserted into a fresh PostgreSQL `market_telemetry_events` table
- **Labeled synthetic evidence** for the externally-produced gates: G5 (multilingual QA), G6 (parity regression), G7 (load/soak). Every JSON file under `data/governance/canary_evidence/jp-2026-04-16/` carries a `_provenance` field declaring it as dress-rehearsal output

When production evidence becomes available, it supersedes the synthetic files and the gate runner can be re-run unchanged.

## Environment

| Item | Value |
|---|---|
| Postgres | `postgres:16` ephemeral container `asia-canary-jp-pg` on port 55435 |
| Schema | head — all migrations applied via `alembic upgrade head` |
| Gate runtime | `backend/scripts/run_market_canary_rehearsal.py` invoking `backend/app/services/governance/launch_gates.py` |
| Seeded telemetry | 4 `universe_drift` rows (worst ratio = 0.012 on JP), 4 markets × 1 `completeness_distribution` row each (JP low-bucket = 0.03; worst overall = 0.04 on TW), plus JP `freshness_lag` and `benchmark_age` rows |
| Evidence dir | `data/governance/canary_evidence/jp-2026-04-16/` |
| Fixed runner clock | `2026-04-16T12:00:00+00:00` |

## Procedure executed

The physical flag-toggle sequence from the HK playbook was not replayed in a live environment because there is no real JP production target inside this repo sandbox. What was executed:

- **Stage 0 (preconditions)**: started a fresh `postgres:16` container and applied migrations to head
- **Stages 1–6 (post-rollout state)**: seeded JP-focused telemetry and attached JP-specific external evidence, with emphasis on `^N225` correctness and higher-volume load behavior
- **Stage 7 (post-canary verdict)**: ran the checked-in harness, rendered the signed artifact trio (`.json`, `.md`, `.sha256`), and verified the sidecar hash with `sha256sum -c`

## Per-gate results

| Gate | Name | Status | Detail |
|---|---|---|---|
| G1 | Schema/Contract Readiness | **pass** | Picks up `asia_v2_e11_st2_migration_rehearsal_report_2026-04-15.md` with no-data-loss assertion |
| G2 | Universe Integrity and Freshness | **pass** | Worst JP universe drift ratio in last 2d = **0.012** (< 0.15 critical) |
| G3 | Benchmark/Calendar Correctness | **pass** | Registry maps US→SPY, HK→^HSI, JP→^N225, TW→^TWII; JP benchmark remains anchored on `^N225` |
| G4 | Fundamentals Data Quality | **pass** | Worst market completeness 0-25 bucket = **0.04** on TW (< 0.50 critical); JP low-bucket remains **0.03** |
| G5 | Multilingual Extraction Quality | **pass** | precision=0.920, recall=0.840, fpr=0.050 (thresholds 0.85/0.75/0.10) - synthetic |
| G6 | US Parity and Non-US Scan Correctness | **pass** | `us_parity_pass=true`, `non_us_correctness_pass=true` - synthetic |
| G7 | Performance and Stability | **pass** | p95=1325ms, failure_rate=0.005, market_isolation=true - synthetic higher-volume JP soak |
| G8 | Observability and Operations Readiness | **pass** | Runbook present; drill `asia_v2_runbook_drill_2026-04-15.md` is **1 day** old (<= 14 day budget) |
| G9 | Rollback Control Validation | **pass** | Flag matrix references all 6 required kill switches |

## Signed artifact

```
data/governance/launch_gates/2026-04-16-pass.json
data/governance/launch_gates/2026-04-16-pass.md
data/governance/launch_gates/2026-04-16-pass.sha256
```

| Hash | Value |
|---|---|
| `content_hash` (semantic, inside JSON) | `3c63d625dfcf74bd3beb2fcd46c0ffdc98c5bee63c72f229a237ad84c7fb0442` |
| File hash (sidecar, sha256sum-compatible) | `d7d47bba3677b43c774b5493d9fc1558fba31c0cc1cef6e3638ccab52e25fb59` |

Verification was performed with `sha256sum -c 2026-04-16-pass.sha256`, which returned `OK`.

## Market-specific notes

JP is the first canary in this sequence explicitly tuned for higher-volume behavior. The synthetic load evidence therefore raises exercised scan volume above the HK rehearsal while staying under the same charter thresholds. The calendar/benchmark failure mode this bead is meant to catch remains the same: any leakage away from `^N225` would fail G3.

## Residual risk

The DB-backed rehearsal path is now revalidated. The remaining dress-rehearsal caveat is external to the runner: G5/G6/G7 still use synthetic evidence files rather than production-generated outputs, which is the same limitation accepted in the HK rehearsal.

## Concrete rerun command

If this dress rehearsal needs to be rerun, use the checked-in harness rather than recreating the seeded DB state by hand:

```bash
docker rm -f asia-canary-jp-pg 2>/dev/null || true
docker run --name asia-canary-jp-pg \
  -e POSTGRES_USER=stockscanner \
  -e POSTGRES_PASSWORD=stockscanner \
  -e POSTGRES_DB=stockscanner_asia_canary_jp \
  -p 55435:5432 \
  -d postgres:16

cd backend
source venv/bin/activate
export DATABASE_URL=postgresql://stockscanner:stockscanner@127.0.0.1:55435/stockscanner_asia_canary_jp
python scripts/run_market_canary_rehearsal.py \
  --market JP \
  --evidence-dir data/governance/canary_evidence/jp-2026-04-16
```

## Defects found during the dress rehearsal

| # | Description | Resolution |
|---|---|---|
| 1 | `run_market_canary_rehearsal.py` originally invoked bare `alembic`, which could pick up the wrong interpreter and fail migrations even when `venv/bin/alembic` succeeded. | Fixed the harness to invoke Alembic via `sys.executable -m alembic` and added regression coverage in `backend/tests/unit/test_run_market_canary_rehearsal.py`. |

## Sign-offs

| Role | Signed at |
|---|---|
| Operator | 2026-04-16 |
| Observer | 2026-04-16 |
| IC | 2026-04-16 |

## Decision

Per the playbook decision tree:

> **PASS** - The JP dress rehearsal completed against real Postgres and produced a signed PASS artifact.

In dress-rehearsal mode the session-hold requirement is informational only. Within this repo's staged rollout workflow, this record supersedes the provisional JP run and is the unblock signal for `asia.11.5`.
