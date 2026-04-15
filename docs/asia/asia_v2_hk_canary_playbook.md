# ASIA v2 HK Canary Rollout Playbook

- Date: 2026-04-15
- Status: Publication artifact for `StockScreenClaude-asia.11.3`
- Scope: enable HK behind feature flags, validate all launch gates, decision PASS / NO_GO / ROLLBACK
- Required by: Launch Gate Charter (G1–G9), Flag Matrix (`asia_v2_flag_matrix_and_rollback_runbook.md`), Operator Runbooks (`asia_v2_operator_runbooks.md`)
- First execution: `asia_v2_hk_canary_execution_2026-04-15.md`

## Purpose

HK is the first ASIA market to be enabled in production. This playbook converts the launch-gate charter into an executable canary procedure: what to do, in what order, with what verification at each step, and where the rollback exits are.

Future canaries (JP per `asia.11.4`, TW per `asia.11.5`) replay the same playbook with their own market substituted.

## Pre-Canary Checklist

Before running this playbook:

- [ ] Every dependency bead in `asia.11` is closed (`bd ready` shows no `asia.11.3`-blocking work).
- [ ] The most recent **migration rehearsal report** (`asia_v2_e*_*_migration_rehearsal_report_*.md`) is dated within 7 days.
- [ ] The most recent **runbook drill record** (`asia_v2_runbook_drill_*.md`) is dated within 14 days (per launch gate G8).
- [ ] External evidence is available for the three external-evidence gates (G5 multilingual QA, G6 parity regression, G7 load/soak).
- [ ] The Operations dashboard (`/operations`) shows zero active critical alerts on US (no rollout while a regression is being investigated on the parity market).

## Roles

| Role | Owner | Responsibilities |
|---|---|---|
| Incident Commander (IC) | `asia-ops` on-call | Decision authority for PASS / NO_GO / ROLLBACK |
| Operator | secondary engineer | Executes flag toggles + verification |
| Observer | `platform-ops` | Watches dashboard, vouches for verification timings |
| Comms | release manager | Posts status to internal channels at each stage |

## Procedure

### Stage 0 — Capture preconditions

```bash
# Snapshot the current launch-gate verdict BEFORE flipping any flag.
python backend/scripts/run_launch_gates.py \
  --evidence "G5=$EVIDENCE_DIR/multilingual_qa.json" \
  --evidence "G6=$EVIDENCE_DIR/parity_regression.json" \
  --evidence "G7=$EVIDENCE_DIR/load_soak.json"
```

Expected exit code: `0` (PASS). If `1` (NO_GO), gather the missing evidence and re-run; if `2` (FAIL), investigate the regression and DO NOT proceed. The signed artifact under `data/governance/launch_gates/<date>-<verdict>.{json,md,sha256}` is the IC's pre-canary go/no-go record.

### Stage 1 — Enable HK at the master + market layer (no subsystem exposure)

```
asia_master_enabled=true
asia_market_hk_enabled=true
```

**Verification (≤ 60s)**: HK appears in the universe selector but every HK-specific subsystem flag remains off. No HK ingestion runs, no HK scans accept requests, no HK extraction enqueues.

**Telemetry expected**: none new — HK is "visible but inert."

### Stage 2 — Enable HK ingestion (write path) with quarantine enforced

```
asia_reconciliation_quarantine_enforced=true   # confirm still on
asia_universe_apply_destructive_enabled=false  # confirm still off
asia_ingestion_HK_enabled=true
```

**Verification (≤ 1 Beat cycle, max 90s per RB-01 budget)**: a `freshness_lag` event for HK appears in `market_telemetry_events` with `source=prices` and a non-zero `symbols_refreshed`. A `universe_drift` event appears with `current_size > 0`.

**Rollback exit**: if drift ratio breaches RB-03 critical (≥ 0.15), execute RB-03 and abort canary. The destructive-write stop is already in place; `asia_ingestion_HK_enabled=false` halts the write path immediately.

### Stage 3 — Enable HK benchmark cache

```
asia_benchmark_cache_HK_enabled=true
```

**Verification (≤ 90s)**: a `benchmark_age` event with `benchmark_symbol="^HSI"` appears for HK. **Critical**: the symbol must be `^HSI`, NOT `SPY`. A SPY entry here is the regression scenario described in RB-02 — execute RB-02 immediately if seen.

### Stage 4 — Enable HK scans (read path) with API exposure off

```
asia_scans_HK_enabled=true
```

**Verification (immediate)**: a scan POST against the HK universe returns 200 (not 4xx). The Operations dashboard now shows HK with non-null gauges for all 5 metric_keys.

### Stage 5 — Enable HK theme extraction

```
asia_themes_HK_enabled=true
```

**Verification (≤ 5 min)**: an `extraction_success` event lands under `SHARED` scope with `payload.language` ∈ {zh, zh-TW} and `payload.provider="minimax"`. Provider must NOT be `groq` (sanctioned-provider contract per CLAUDE.md / RB-05).

### Stage 6 — Enable HK in UI exposure

```
asia_ui_exposure_HK_enabled=true
```

**Verification (immediate)**: `/api/v1/stocks/markets` includes HK in the response. The frontend universe selector renders HK.

### Stage 7 — Capture post-canary verdict

```bash
python backend/scripts/run_launch_gates.py \
  --evidence "G5=$EVIDENCE_DIR/multilingual_qa.json" \
  --evidence "G6=$EVIDENCE_DIR/parity_regression.json" \
  --evidence "G7=$EVIDENCE_DIR/load_soak.json"
```

This artifact is the **canary execution evidence** the IC files for the HK→JP progression decision (`asia.11.4` cannot start until this is PASS).

## Decision Tree

After Stage 7:

| Verdict | Decision | Next |
|---|---|---|
| **PASS** | Canary succeeded | Hold HK enabled for ≥ 2 market sessions; if dashboards stay quiet, mark canary final and unblock JP (`asia.11.4`). |
| **NO_GO** | Missing evidence — DO NOT proceed | Determine which gate reported MISSING_EVIDENCE; produce that evidence and re-run from Stage 7. |
| **FAIL** | A gate reports a real breach | Execute the rollback playbook below. Postmortem within 24h. |

## Rollback Playbook (if FAIL or in-flight breach)

Reverse order of Stages 1–6 (see `asia_v2_flag_matrix_and_rollback_runbook.md` "Standard Rollback Order"):

1. `asia_ui_exposure_HK_enabled=false` — hide HK from users first
2. `asia_themes_HK_enabled=false`
3. `asia_scans_HK_enabled=false`
4. `asia_benchmark_cache_HK_enabled=false`
5. `asia_ingestion_HK_enabled=false`
6. (escalation) `asia_market_hk_enabled=false` — full HK shutdown
7. (cross-market regression only) `asia_master_enabled=false`

After rollback:

- Run `run_launch_gates.py` again — the verdict should return to the pre-canary state.
- File a postmortem using the SEV-1/2/3 templates in `asia_v2_operator_runbooks.md` "Incident Templates".
- The postmortem becomes attached evidence on this bead's close note.

## Acceptance for `asia.11.3`

This bead is satisfied when **either**:

- A signed launch-gate artifact with `verdict=pass` exists under `data/governance/launch_gates/`, **and** the canary ran end-to-end (all 7 stages executed against staging or production), **and** an execution record (`asia_v2_hk_canary_execution_<date>.md`) records the run; **OR**
- A rollback was executed per the playbook **and** a postmortem document exists under `docs/asia/`.

Both outcomes are valid completions — what matters is the documented evidence trail.
