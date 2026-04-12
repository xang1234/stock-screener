# ASIA v2 JP Ingestion Adapter Notes

- Date: 2026-04-12
- Issue: `StockScreenClaude-asia.4.2`

## Exchange Mapping Assumptions

JP universe ingestion normalizes source exchange aliases to a canonical value:

- `TSE` -> `XTKS`
- `JPX` -> `XTKS`
- `XTKS` -> `XTKS`

The adapter writes canonical exchange `XTKS` into `stock_universe.exchange` for diff-safe snapshots.

## Symbol Format Assumptions

The JP adapter accepts source symbol variants and canonicalizes to `*.T` via SecurityMaster:

- `7203`
- `7203.T`
- `JPX:7203`
- `7203.JP`

Canonical output is `7203.T` with `market=JP`, `currency=JPY`, `timezone=Asia/Tokyo`, and `local_code=7203`.

## Lifecycle Edge Cases

To reduce churn during listing transfers or source format drift:

- Alpha-suffixed local codes are accepted by policy (`[0-9]{3,5}[A-Z]?`).
- Canonical rows include deterministic `lineage_hash` and `row_hash` values.
- Dedupe is deterministic by canonical symbol and source ordering, so repeated snapshots are diff-stable.

These rules are intentionally strict about invalid non-JP formats and surface rejected rows explicitly for operator review.

