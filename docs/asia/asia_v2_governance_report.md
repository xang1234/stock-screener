# ASIA v2 Weekly Telemetry Governance Report

- Date: 2026-04-15
- Status: Publication artifact for `StockScreenClaude-asia.10.4`
- Scope: scheduled Celery job + on-disk report format + verification procedure
- Required by: Launch Gate **G8** (Observability and Operations Readiness) in `asia_v2_launch_gate_charter.md` — the weekly report is the long-form evidence that complements the real-time alerts/dashboard

## Purpose

The raw telemetry event log (`market_telemetry_events`, bead 10.1) has a 15-day retention window. Alerts (bead 10.2) fire on threshold breaches but don't record the between-breach state. The weekly governance report is the **long-lived summary**: it aggregates 7 days of per-market telemetry into a tamper-evident artifact that persists indefinitely, even after the raw events expire.

## Schedule

- **Task name**: `app.tasks.telemetry_tasks.weekly_telemetry_audit`
- **Cadence**: Sunday 05:00 ET (via Celery Beat in `backend/app/celery_app.py`)
- **Ordering**: runs after the Sunday 02:00 full refresh, 03:00 universe refresh, 03:30 taxonomy, and 04:00 consolidation tasks so the final telemetry of the week is already on disk
- **Queue**: default `celery` queue (the audit is DB-only — it doesn't fetch external data)

## Report Path

Artifacts are written to:

```
<project-root>/data/governance/telemetry_audit/
├── 2026-04-19.json      # canonical report (machine-readable)
├── 2026-04-19.md        # human-readable rendering
└── 2026-04-19.sha256    # SHA-256 of the JSON file, sha256sum format
```

Override with the `TELEMETRY_AUDIT_REPORT_DIR` environment variable on the Celery worker, or pass `output_dir=...` when invoking the task manually.

## Artifact Contract

### JSON (authoritative)

Top-level fields:

| Field | Type | Meaning |
|---|---|---|
| `report_schema_version` | int | Bump on non-additive changes to this contract |
| `payload_schema_version` | int | Matches `SCHEMA_VERSION` in `backend/app/services/telemetry/schema.py` at generation time |
| `generated_at` | ISO-8601 UTC | Wall-clock time the report was produced |
| `window_start` / `window_end` | ISO-8601 UTC | Exactly `AUDIT_WINDOW_DAYS` (7) apart |
| `markets` | string[] | `["US", "HK", "JP", "TW", "SHARED"]` at the time of run |
| `metrics` | array of objects | One entry per (market, metric_key) with event_count + rollup |
| `alerts` | array of objects | One entry per market with opened/closed/still_active counts |
| `thresholds_snapshot` | nested dict | Point-in-time copy of `THRESHOLDS` |
| `owners_snapshot` | dict | Point-in-time copy of `OWNERS` |
| `content_hash` | string | SHA-256 of this JSON with `content_hash` nulled out |

### Markdown (operator-facing)

Human-readable rendering of the JSON. Prints the content hash at both top and bottom of the file — a tamper attempt that only rewrites the visible header would still leave a mismatched footer.

### SHA-256 file

Single line, `sha256sum` format:

```
1d55e3855759bcea5f7c118f8c9185aa6f58633b15f49072ccb69324d8275156  2026-04-19.json
```

## Verification

From the report directory:

```bash
cd data/governance/telemetry_audit/
sha256sum -c 2026-04-19.sha256
```

Programmatic verification (Python):

```python
import hashlib, json
with open("2026-04-19.json") as f:
    blob = json.load(f)
expected = blob["content_hash"]
blob["content_hash"] = None
recomputed = hashlib.sha256(
    json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str).encode()
).hexdigest()
assert expected == recomputed, "Report has been tampered with or is corrupt"
```

## Rollup Semantics (per metric_key)

| Metric | Rollup fields |
|---|---|
| `freshness_lag` | `freshness_at_report_seconds` (window_end − latest refresh), `max_gap_between_refreshes_seconds` (longest consecutive-refresh gap — catches pipeline stalls that recovered), `refresh_events_with_symbols` |
| `universe_drift` | `max_drift_ratio` (\|delta\|/prior_size), `cumulative_abs_delta` |
| `benchmark_age` | `latest_benchmark_symbol`, `latest_warmed_at_epoch`, `implied_age_seconds` |
| `extraction_success` | `overall_total`, `overall_success`, `overall_success_ratio`, `by_language` |
| `completeness_distribution` | `first_snapshot_low_bucket_ratio`, `last_snapshot_low_bucket_ratio`, `low_bucket_ratio_delta`, `last_snapshot_symbols_total` |

The `low_bucket_ratio_delta` for completeness is the **governance regression signal**: a positive delta means the fraction of symbols with sub-25% provenance grew over the week.

## Governance Use

1. **Launch-gate evidence**: the most recent report is attached to the launch-gate evidence bundle for Charter G8.
2. **Weekly review**: on-call from each owner team reviews the Markdown rendering. A `still_active` count > 0 at week close triggers escalation per `asia_v2_operator_runbooks.md`.
3. **Historical audit**: each report is a point-in-time snapshot of thresholds + owners + rollups. Reviewers looking at a 3-month-old report can re-interpret it against **that report's** threshold snapshot, not today's.

## Change Control

- Adding a new metric: extend `_METRIC_KEYS` in `weekly_audit.py` and add a branch to `_rollup_for_metric`. Do not bump `REPORT_SCHEMA_VERSION` (this is additive).
- Renaming/removing a rollup field: bump `REPORT_SCHEMA_VERSION` and note the break in this document.
- Changing the hash algorithm: bump `REPORT_SCHEMA_VERSION` and add a migration note so old reports remain verifiable with the old algorithm.
