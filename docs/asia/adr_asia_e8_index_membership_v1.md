# ADR ASIA-E8: Asia Index Membership Storage and Seed Workflow (v1)

- Date: 2026-04-14
- Status: Accepted
- Scope: `StockScreenClaude-asia.8` follow-ups (beads `7hwc` + `mnpo`)
- Supersedes: N/A

## Context

Before this change, the scan universe supported one index (`SP500`) via a
boolean column `stock_universe.is_sp500`. The INDEX branch of
`universe_resolver` hardcoded `sp500_only=True` regardless of which
`IndexName` value the client passed, silently returning SP500 symbols for
any requested index. With Asia expansion (HK, JP, TW) we need HSI, Nikkei
225, and a Taiwan index to be first-class scan scopes.

## Decisions

### D1: Separate membership table, not parallel boolean columns

Adding `is_hsi`, `is_nikkei225`, `is_taiex` columns on `stock_universe`
would have mirrored the existing `is_sp500` pattern. Rejected because:

- Every new index (hypothetical FTSE 100, KOSPI 200, STOXX 600) would
  require a schema migration + ORM change.
- The universal "which indices contain this symbol?" query becomes a
  column scan over booleans; a membership table with an index on
  `symbol` answers it with a bounded B-tree lookup.

Chosen: new `stock_universe_index_membership(symbol, index_name,
as_of_date, source)` with `UNIQUE(symbol, index_name)` and composite index
on `(index_name, symbol)`. SP500 is **kept** on `is_sp500` (not migrated)
for backward compatibility with existing code paths that read the column
directly.

### D2: TAIEX narrowed to TAIEX-50 (top 50 by weight)

The TAIEX (Taiwan Weighted Index) covers ~900 listed stocks — near-total
Taiwan market coverage. Exposing it as an "index" scope would be
indistinguishable from the existing `{"type":"market","market":"TW"}`
scope and dilute the index concept (users selecting "TAIEX" expect a
curated top-N, not "every listed Taiwanese stock").

Chosen: use the `IndexName.TAIEX` enum value to hold **TAIEX-50** — the
top 50 constituents by weight. Frontend label is "TAIEX 50" to prevent
confusion. If a full-TAIEX scan is ever needed, add a new
`IndexName.TAIEX_FULL` enum value at that point rather than retrofitting
semantics.

### D3: CSV-driven seed workflow, not live scraping

HSI, Nikkei 225, and TAIEX-50 rebalance quarterly. A live adapter
scraping index provider websites is more fragile (CAPTCHA, rate limits,
schema changes) than the quarterly maintenance load justifies.

Chosen: check seed CSVs into `backend/app/data/` as the authoritative
snapshot. An idempotent management script
(`scripts/seed_index_memberships.py`) loads them via
`app.services.index_membership_seeder.seed_from_csv`. Operator refreshes
the CSV on rebalance day and re-runs the script.

### D4: Partial seed-v1 CSVs acceptable for initial ship

The seed CSVs checked in at this bead are best-effort snapshots from the
code author's May 2025 knowledge cutoff. They cover ~70/80 HSI
constituents, ~63/225 Nikkei 225 constituents, and ~42/50 TAIEX-50
constituents. See `backend/app/data/README.md` for the refresh process.

This is sufficient for:

- End-to-end feature validation (the vertical slice runs).
- Staging/demo deployments where stake holders can test index scans.
- Integration-test coverage (seeded fixtures exist).

It is **not** sufficient for production without a refresh from each
index provider's official list. The README makes this explicit.

### D5: Fail-closed on unseeded / unknown indices

`StockUniverseService.get_active_symbols` returns `[]` (with a
`logger.warning`) when asked for an `index_name` not in the `IndexName`
enum or whose membership table is empty. The alternative — falling
through to "all active symbols" — would silently leak a whole-market
scan when membership data hadn't loaded yet.

## Consequences

### Positive

- One schema change, future indices require only data.
- Resolver dispatch is enum-driven and type-safe.
- Seed workflow is observable: the script prints `added/updated/
  unchanged` counts and is idempotent, so operators can re-run safely.

### Negative / accepted

- Quarterly rebalances are a manual operator task until a live
  ingestion adapter is built. Follow-up bead: "T8 HSI/Nikkei/TAIEX
  automated ingestion adapter" (not yet filed — file when the manual
  cadence becomes painful).
- The seed script does not currently prune rows for constituents that
  drop out of the new CSV. Quarterly churn is low (1–3 changes per
  rebalance), but after a few cycles the DB will hold stale
  memberships. Mitigation documented in `app/data/README.md`; follow-up
  if painful.
- Seed-v1 CSV accuracy is partial. Stakeholders reading scan results
  need to know the constituent list is a snapshot; the README carries
  this warning.

## References

- `backend/alembic/versions/20260414_0011_add_stock_universe_index_membership.py`
- `backend/app/models/stock_universe.py` (`StockUniverseIndexMembership`)
- `backend/app/schemas/universe.py` (`IndexName` enum)
- `backend/app/services/index_membership_seeder.py`
- `backend/app/data/README.md` (operator runbook)
- Closed beads: `StockScreenClaude-7hwc` (schema + resolver),
  `StockScreenClaude-mnpo` (seed data + frontend + this ADR)
