# Index Constituent Seed Data

This directory holds CSV snapshots of index constituent lists used to seed
`stock_universe_index_membership` via `scripts/seed_index_memberships.py`.

## Status: seed-v1 (partial, best-effort)

**These CSVs are NOT authoritative.** They are a best-effort snapshot of
well-known constituents from May 2025 (the author's knowledge cutoff at
seed time), included so the multi-market scan feature has end-to-end
coverage for testing, demo, and staging deployments.

Before production use:

1. Refresh each CSV from an official source:
   - HSI — [Hang Seng Indexes Company](https://www.hsi.com.hk/)
   - Nikkei 225 — [Nikkei index official list](https://indexes.nikkei.co.jp/)
   - TAIEX-50 — Taiwan Stock Exchange FTSE TWSE index constituents (top 50 by weight)
2. Update the `as_of_date` column header comment and re-run the seed
   script with the new `--as-of` value.
3. Run against staging first; verify row counts match the official list
   size (HSI ≈ 80, Nikkei 225 = 225, TAIEX-50 = 50).

## CSV schema

Every constituent CSV uses the same two-column shape:

```csv
symbol,name
0700.HK,Tencent Holdings
0005.HK,HSBC Holdings
...
```

- `symbol` is the canonical suffixed form stored in `stock_universe.symbol`
  (`.HK` for Hong Kong, `.T` for Tokyo, `.TW` for Taiwan).
- `name` is the human-readable company name. It's stored in
  `stock_universe.name` when the symbol is ingested separately; the seed
  script ignores it but keeps the column so operators can review the CSV
  without pulling up the DB.
- CSV row order is informational. The seed script walks rows in file order but
  the `stock_universe_index_membership` table has no ordinal column, so
  downstream queries do not preserve CSV order. For TAIEX-50 the rows are
  listed top-down by index weight as a human reference, but scans return
  results sorted by market-cap, not constituent weight.

## Refresh workflow (quarterly)

The Asia indices rebalance quarterly. The recommended workflow:

1. Download the new constituent list, convert to the 2-column CSV format.
2. Check the new CSV in as `<index>_constituents_<YYYY-MM>.csv` (e.g.
   `hsi_constituents_2026-06.csv`). Keep the previous snapshot for audit.
3. Run:
   ```bash
   python scripts/seed_index_memberships.py \
     --csv app/data/hsi_constituents_2026-06.csv \
     --index HSI \
     --as-of 2026-06-01
   ```
4. Validate row counts against the official list length (HSI ≈ 80 etc.).
   `seed_from_csv` is idempotent; re-running is safe.

> **Known gap:** the seed script does not currently delete rows whose
> symbols drop out of the new CSV (ex-constituents stay stale). This is
> acceptable for the initial seed because the membership table is empty;
> for quarterly rebalances with real churn, add a "prune missing" step
> or clear the index before reseeding. Tracked as a follow-up if needed.

## Live ingestion (future)

A live adapter that scrapes each index provider's website and writes
directly to the membership table is explicitly out of scope for the
seed-v1 bead. File a new bead (`T8 HSI automated ingestion adapter`
etc.) if the manual quarterly refresh proves insufficient.
