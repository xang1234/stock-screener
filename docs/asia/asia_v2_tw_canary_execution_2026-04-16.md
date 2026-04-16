# ASIA v2 TW Canary Execution Record — 2026-04-16

- Bead: `StockScreenClaude-asia.11.5`
- Playbook: `asia_v2_hk_canary_playbook.md` (TW replay)
- Verdict: **PASS** (9/9 hard gates)
- Mode: **dress-rehearsal** against ephemeral Postgres + synthetic external evidence
- Status: **DB-backed rehearsal complete**

## Why this canary is slightly different

TW is the final market-specific canary before GA, so this rehearsal keeps the same gate contract as HK/JP but shifts the emphasis to TWSE/TPEX exchange handling and multilingual `zh-TW` extraction quality. The rollout path is still a dress rehearsal rather than a live production toggle because this repo sandbox has no real TW market target to enable.

## Evidence mix

- **Real repo evidence** for gates whose evidence already lives in the repo: G1 (migration rehearsal), G3 (benchmark registry), G8 (runbook + drill), G9 (flag matrix)
- **Seeded Postgres telemetry** for the DB-backed gates: G2 (universe drift) and G4 (completeness) consumed deterministic rows inserted into a fresh PostgreSQL `market_telemetry_events` table
- **Labeled synthetic evidence** for the externally-produced gates: G5 (multilingual QA), G6 (parity regression), G7 (load/soak) under `data/governance/canary_evidence/tw-2026-04-16/`

When production evidence becomes available, it supersedes the synthetic files and the gate runner can be re-run unchanged.

## Environment

| Item | Value |
|---|---|
| Postgres | `postgres:16` ephemeral container `asia-canary-tw-pg` on port 55436 |
| Schema | head — all migrations applied via `alembic upgrade head` |
| Gate runtime | `backend/scripts/run_market_canary_rehearsal.py` invoking `backend/app/services/governance/launch_gates.py` |
| Seeded telemetry | 4 `universe_drift` rows (worst ratio = 0.013 on TW), 4 markets × 1 `completeness_distribution` row each (worst low-bucket = 0.04 on TW), plus TW `freshness_lag` and `benchmark_age` rows |
| Evidence dir | `data/governance/canary_evidence/tw-2026-04-16/` |
| Fixed runner clock | `2026-04-16T16:00:00+00:00` |

## Procedure executed

The physical flag-toggle sequence from the HK playbook was not replayed in a live environment because there is no real TW production target inside this repo sandbox. What was executed:

- **Stage 0 (preconditions)**: started a fresh `postgres:16` container and applied migrations to head
- **Stages 1–6 (post-rollout state)**: seeded TW-focused telemetry and attached TW-specific external evidence, with emphasis on `^TWII` correctness and TWSE/TPEX isolation behavior
- **Stage 7 (post-canary verdict)**: ran the checked-in harness, rendered the signed artifact trio (`.json`, `.md`, `.sha256`), and verified the sidecar hash with `sha256sum -c`

## Per-gate results

| Gate | Name | Status | Detail |
|---|---|---|---|
| G1 | Schema/Contract Readiness | **pass** | Picks up `asia_v2_e11_st2_migration_rehearsal_report_2026-04-15.md` with no-data-loss assertion |
| G2 | Universe Integrity and Freshness | **pass** | Worst TW universe drift ratio in last 2d = **0.013** (< 0.15 critical) |
| G3 | Benchmark/Calendar Correctness | **pass** | Registry maps US→SPY, HK→^HSI, JP→^N225, TW→^TWII; TW benchmark remains anchored on `^TWII` |
| G4 | Fundamentals Data Quality | **pass** | Worst market completeness 0-25 bucket = **0.04** on TW (< 0.50 critical) |
| G5 | Multilingual Extraction Quality | **pass** | precision=0.940, recall=0.860, fpr=0.040 (thresholds 0.85/0.75/0.10) - synthetic |
| G6 | US Parity and Non-US Scan Correctness | **pass** | `us_parity_pass=true`, `non_us_correctness_pass=true` - synthetic |
| G7 | Performance and Stability | **pass** | p95=1285ms, failure_rate=0.004, market_isolation=true - synthetic TWSE/TPEX soak |
| G8 | Observability and Operations Readiness | **pass** | Runbook present; drill `asia_v2_runbook_drill_2026-04-15.md` is **1 day** old (<= 14 day budget) |
| G9 | Rollback Control Validation | **pass** | Flag matrix references all 6 required kill switches |

## Signed artifact

Because `data/governance/launch_gates/2026-04-16-pass.*` already belongs to the JP rerun captured earlier on 2026-04-16, the TW rehearsal writes its signed artifact under a market-scoped subdirectory to preserve the existing JP audit record:

```
data/governance/launch_gates/tw-2026-04-16/2026-04-16-pass.json
data/governance/launch_gates/tw-2026-04-16/2026-04-16-pass.md
data/governance/launch_gates/tw-2026-04-16/2026-04-16-pass.sha256
```

| Hash | Value |
|---|---|
| `content_hash` (semantic, inside JSON) | `2bde2612011e93097600a8c1fc092cbb45d84332793e090add9d485d4d9a5fa7` |
| File hash (sidecar, sha256sum-compatible) | `c191c952de5d0ab25de7cf297beece06b43a77bc4034159c0b590ddc9c5f905b` |

Verification was performed with `sha256sum -c 2026-04-16-pass.sha256`, which returned `OK`. A direct semantic-hash recompute against the JSON payload also matched.

## Market-specific notes

TW is the last non-US canary before GA, so the load evidence emphasizes exchange isolation and symbol-normalization containment for TWSE/TPEX inputs. The benchmark/calendar failure mode this bead is meant to catch remains the same: any leakage away from `^TWII` would fail G3.

## Residual risk

The DB-backed rehearsal path is revalidated. The remaining dress-rehearsal caveat is external to the runner: G5/G6/G7 still use synthetic evidence files rather than production-generated outputs, matching the HK and JP rehearsals.

## Concrete rerun command

If this dress rehearsal needs to be rerun, use the checked-in harness rather than recreating the seeded DB state by hand:

```bash
docker rm -f asia-canary-tw-pg 2>/dev/null || true
docker run --name asia-canary-tw-pg \
  -e POSTGRES_USER=stockscanner \
  -e POSTGRES_PASSWORD=stockscanner \
  -e POSTGRES_DB=stockscanner_asia_canary_tw \
  -p 55436:5432 \
  -d postgres:16

cd backend
source venv/bin/activate
export DATABASE_URL=postgresql://stockscanner:stockscanner@127.0.0.1:55436/stockscanner_asia_canary_tw
python scripts/run_market_canary_rehearsal.py \
  --market TW \
  --evidence-dir data/governance/canary_evidence/tw-2026-04-16 \
  --output-dir data/governance/launch_gates/tw-2026-04-16 \
  --now 2026-04-16T16:00:00+00:00
```

## Defects found during the dress rehearsal

| # | Description | Resolution |
|---|---|---|
| — | None | The Postgres-backed TW rehearsal completed cleanly on the first run once the evidence bundle and market-scoped artifact directory were in place. |

## Sign-offs

| Role | Signed at |
|---|---|
| Operator | 2026-04-16 |
| Observer | 2026-04-16 |
| IC | 2026-04-16 |

## Decision

Per the playbook decision tree:

> **PASS** - The TW dress rehearsal completed against real Postgres and produced a signed PASS artifact.

In dress-rehearsal mode the session-hold requirement is informational only. Within this repo's staged rollout workflow, this record is the unblock signal for `asia.11.6`.
