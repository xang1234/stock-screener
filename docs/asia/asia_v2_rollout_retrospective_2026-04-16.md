# ASIA v2 Rollout Retrospective — 2026-04-16

- Bead: `StockScreenClaude-asia.11.6`
- Scope: retrospective for the staged HK → JP → TW rollout workflow
- Outcome: **launch workflow completed; follow-up backlog created**

## What went well

1. The consolidated launch-gate runner gave a consistent contract across HK, JP, and TW.
2. The Postgres-backed canary harness made reruns cheap once the harness existed.
3. Benchmark registry and market calendar abstractions held across all supported markets once the Sunday session bug was fixed.
4. The signed artifact pattern (`json` + `md` + `sha256` + `content_hash`) made every canary verdict auditable.
5. Runbook and telemetry work from E10 paid off: there was enough observability structure to reason about failures without inventing ad hoc checks.

## What went wrong

1. JP originally had to ship as a provisional synthetic-only rehearsal because the environment lacked Docker/Postgres access.
2. The canary harness originally invoked bare `alembic`, which picked up the wrong interpreter and broke migrations even though the project venv was healthy.
3. Same-day artifact naming (`YYYY-MM-DD-<verdict>`) collided once multiple markets were rehearsed on 2026-04-16.
4. Execution records were manually assembled, which led to copy/paste drift risk in hashes and example paths.
5. The final data-availability gate looked complete in subsystem docs/tests, but it had no formal evidence record until this session.
6. Calendar verification surfaced a real Sunday regression in `MarketCalendarService.last_completed_trading_day()` that would have made the final GA evidence dishonest if left unfixed.

## Concrete fixes made during rollout

- Added `backend/scripts/run_market_canary_rehearsal.py` so HK/JP/TW can be rerun through the same Postgres-backed path.
- Fixed the harness to run Alembic through `sys.executable -m alembic`.
- Reran and documented HK, JP, and TW dress rehearsals with signed artifacts.
- Fixed `MarketCalendarService.last_completed_trading_day()` so non-session dates use `date_to_session(..., direction="previous")` under `exchange_calendars`.
- Converted the final data-availability gate into an explicit dated evidence record.

## Lessons

1. Any launch artifact intended to be market-specific must carry explicit market identity in both filename semantics and report schema.
2. Audit records should be machine-derived whenever possible; manual Markdown summaries are too easy to let drift from the signed artifact.
3. Final signoff should not be treated as documentation-only work. Running the evidence suite late found a real calendar defect.
4. Staged rollout tasks need their dependency graph to reflect every actual hard gate; otherwise GA tasks appear ready before the final verification record exists.

## Follow-up beads created

| Issue | Why it exists |
|---|---|
| `StockScreenClaude-42lw` | Remove same-day artifact collisions without relying on ad hoc subdirectories |
| `StockScreenClaude-0j1p` | Generate execution records directly from harness output to remove manual drift |
| `StockScreenClaude-pxyo` | Add first-class target-market metadata to launch-gate artifacts |

## Recommendation

Keep the HK/JP/TW dress-rehearsal artifacts and this retrospective as the baseline evidence set for any future ASIA extension. New markets should not bypass the harness, the signed artifact contract, or the data-availability launch gate.
