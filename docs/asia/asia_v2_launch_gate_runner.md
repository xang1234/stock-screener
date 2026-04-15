# ASIA v2 Launch-Gate Runner

- Date: 2026-04-15
- Status: Publication artifact for `StockScreenClaude-asia.11.1`
- Scope: deterministic gate runner that aggregates the 9 gates defined in `asia_v2_launch_gate_charter.md` into a single signed pass/fail artifact
- Blocks: `asia.11.2` (migration rehearsal), `asia.11.7` (data-availability gate), all canary beads

## Purpose

Before each canary stage (HK → JP → TW → GA), a reviewer must hold a single artifact that says go/no-go. The runner produces that artifact with cryptographic tamper-evidence so the decision trail is auditable weeks later.

Every gate is deterministic — no subjective judgment at decision time. Subjective judgment lives only in the threshold values (defined in the charter).

## Running the runner

```bash
cd backend
source venv/bin/activate
python scripts/run_launch_gates.py [options]
```

### Options

| Flag | Purpose |
|---|---|
| `--evidence GATE_ID=PATH` | Attach external evidence for G5/G6/G7. Repeatable. |
| `--output-dir PATH` | Override output directory (default: `data/governance/launch_gates/`). |
| `--no-db` | Skip DB-backed gates (G2, G4). They report `MISSING_EVIDENCE`. |
| `--execution-mode MODE` | Optional provenance label embedded into the report (for example `synthetic_seeded_harness`). |
| `--provenance-note TEXT` | Optional human-readable provenance note embedded into the report and markdown. |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | **PASS** — every hard gate passed. Proceed with canary. |
| `1` | **NO_GO** — at least one hard gate `MISSING_EVIDENCE`. Collect the missing evidence and re-run; do not proceed. |
| `2` | **FAIL** — at least one hard gate `FAIL`. Investigate and fix before re-running; do not proceed. |

The `FAIL` vs `NO_GO` distinction is deliberate. A `FAIL` is a regression; a `NO_GO` is missing paperwork. Treating them differently prevents incidents where an operator re-runs with partial evidence and mistakes "we didn't measure it" for "we measured and it passed."

## Gate coverage

| Gate | Name | Evidence source | Check |
|---|---|---|---|
| G1 | Schema/Contract Readiness | `docs/asia/asia_v2_e2_st3_t2_migration_rehearsal_report_*.md` | Report exists with no-data-loss assertion or ≥3 Success rows |
| G2 | Universe Integrity and Freshness | `market_telemetry_events` DB table | Worst `universe_drift` ratio in last 2d < 0.15 |
| G3 | Benchmark/Calendar Correctness | `benchmark_registry_service` | Every ASIA market maps to its index symbol (no SPY leakage) |
| G4 | Fundamentals Data Quality | `market_telemetry_events` DB table | Every market's latest `completeness_distribution` has `0-25` bucket < 0.50 |
| G5 | Multilingual Extraction Quality | External JSON via `--evidence G5=...` | precision ≥ 0.85, recall ≥ 0.75, false-positive-rate ≤ 0.10 |
| G6 | US Parity and Non-US Scan Correctness | External JSON via `--evidence G6=...` | Both `us_parity_pass` and `non_us_correctness_pass` true |
| G7 | Performance and Stability | External JSON via `--evidence G7=...` | p95 ≤ 1500ms, failure rate ≤ 0.01, `market_isolation_pass` true |
| G8 | Observability and Operations Readiness | `docs/asia/asia_v2_operator_runbooks.md` + `asia_v2_runbook_drill_*.md` | Runbook exists, most recent drill ≤ 14 days old |
| G9 | Rollback Control Validation | `docs/asia/asia_v2_flag_matrix_and_rollback_runbook.md` | Document references all 6 required kill-switch flags |

## External evidence file formats

### G5 — Multilingual QA harness

```json
{
  "precision": 0.90,
  "recall": 0.80,
  "false_positive_rate": 0.05
}
```

### G6 — Parity regression

```json
{
  "us_parity_pass": true,
  "non_us_correctness_pass": true
}
```

### G7 — Performance/stability

```json
{
  "scan_create_p95_ms": 1200,
  "scan_failure_rate": 0.005,
  "market_isolation_pass": true
}
```

The runner only reads these fields; extra keys are ignored. Producers of these reports are free to include additional telemetry for audit.

## Artifact path

```
data/governance/launch_gates/
├── 2026-04-19-pass.json      # canonical report (machine-readable)
├── 2026-04-19-pass.md        # human-readable rendering
└── 2026-04-19-pass.sha256    # SHA-256 of the .json file (sha256sum format)
```

Filename stem is `YYYY-MM-DD-<verdict>` so a directory listing shows the outcome without opening any file.

## Verification (dual-hash contract)

Same pattern as the weekly governance report (`asia_v2_governance_report.md`):

**File integrity** — guards against truncation/bit-rot:

```bash
cd data/governance/launch_gates/
sha256sum -c 2026-04-19-pass.sha256
```

**Data integrity** — guards against semantic tampering where both the JSON and the sidecar are rewritten:

```python
import hashlib, json
with open("2026-04-19-pass.json") as f:
    blob = json.load(f)
expected = blob["content_hash"]
blob["content_hash"] = None
recomputed = hashlib.sha256(
    json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str).encode()
).hexdigest()
assert expected == recomputed, "Report data has been semantically altered"
```

## Relationship to the charter

The runner enforces the charter; it is not the charter. When the charter's thresholds change (e.g. moving from p95 1500ms to p95 1200ms), edit the charter AND the gate check in `backend/app/services/governance/launch_gates.py` in the same PR. The test suite re-asserts the thresholds so a silent drift would surface in CI.

## Change control

- Every `--evidence` schema above has a matching gate check in `launch_gates.py`. Adding or renaming a field requires both sides.
- `REPORT_SCHEMA_VERSION` (in `launch_gates.py`) must bump when the output JSON shape changes in a non-additive way.
- The `CHARTER_VERSION` constant must match the charter doc's version header so reports are re-interpretable against their point-in-time charter.
