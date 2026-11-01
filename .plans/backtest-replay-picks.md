# Backtest replay picks

## Problem

The Backtest page (`/validation`) scores published scan picks. Published runs start on 2026-10-09, so the 20- and 60-session horizons have no samples until 2026-11-10 and 2027-01-08, and the 5-session sample is about 30 picks per market. The page cannot say anything about the screener yet.

## What can be replayed honestly

Research on `origin/main` @ `4eff4c3f` found:

- Prices are cached back to about 2024-10 (US, measured).
- Point-in-time RS ratings are stored in `market_rs_runs` and `stock_rs_snapshots` from 2026-04-06 (US, 145 runs), 2026-04-08 (HK), and 2026-04-17 (JP, KR). Earlier US runs failed with an empty point-in-time universe.
- The live composite averages six screeners. CANSLIM, IPO and custom need fundamentals, which have no history. Minervini needs only prices, the benchmark and RS.
- `ScanOrchestrator.scan_stock_multi(..., pre_fetched_data=StockData(...), market_rs_resolution=...)` scores one symbol from frames it is given and makes no provider call (measured with a raising stub, 800 calls).
- Universe membership before 2026-10-05 cannot be reconstructed. The stored RS runs used the current universe, and the price cache only holds symbols that are in today's universe. A replay is therefore survivor-biased.

## Decision

A replay pick is a separate source, never mixed with published picks:

> On session D of market M, among the symbols in M's completed Market RS run for D, excluding funds, score Minervini on prices and benchmark sliced at D with RS from that run. Keep symbols the Minervini screener passes. Take the top 10 by Minervini score, ties by symbol.

| Choice | Default | Alternative the owner can ask for |
|---|---|---|
| Definition | Minervini template, top 10 by Minervini score | Add volume breakthrough as a second price-only score (cheap) or setup engine (about 10x cost) |
| Window | Every date with a completed Market RS run, per market | Relax point-in-time RS to reach back to about 2025-10 over the current universe |
| Survivor bias | Accept and disclose on the page and in the guide | none |
| Markets | Any market with completed RS runs (US, HK, JP, KR today) | none |

Replay dates that also have published runs (2026-10-09 onward) are kept. They let a reader compare the replay definition with the live one on the same days.

## Shape

- **Storage.** `backtest_replay_picks(market, as_of_date, rank, symbol, minervini_score, rs_rating, stage, definition_version, market_rs_run_id, created_at)`, unique on `(market, as_of_date, definition_version, rank)`. Picks only, not every scored row.
- **Runner.** `backend/app/scripts/replay_scan_picks.py --market M [--start D] [--end D] [--dry-run]`. It loads each symbol's OHLCV once from `stock_prices`, slices it per date, and never goes through `DataPreparationLayer` or the price cache service. One date's picks are written in one transaction that replaces that date's rows, so reruns converge and an interrupted run resumes by date. A date where any symbol errors, or where no benchmark candidate has a bar, writes nothing and fails the run.
- **Backtest page.** A new source kind `replay_pick` reads the table and feeds the existing outcome calculator unchanged. The page shows it as "Replay (Minervini)" with a note that it is price-only and survivor-biased.

## Mid-session behaviour

The runner only replays dates that have a completed Market RS run. Those dates are completed sessions, so the runner never reads a partial bar for D. It never fetches from a provider. The page's outcome rules are unchanged: windows cap at the last completed session and pending covers windows not closed yet.

## Cost

Measured on 2026-11-01: a US session scores about 5.4k non-fund symbols in about 30 s on one process, after about 25 s loading prices once, so all 145 US sessions take about 1 to 1.5 hours. The owner runs it, one market at a time and from a worker container, because it writes to the live database.
