"""Backfill setup_engine payloads into existing feature store rows.

Historical StockFeatureDaily rows predate the Setup Engine integration and
lack the ``setup_engine`` key in ``details_json``.  This script computes
setup_engine payloads using historical price data and surgically merges them
into existing rows without disturbing other screener data.

Usage:
    cd backend
    source venv/bin/activate

    # Preview what would be backfilled
    python scripts/backfill_setup_engine.py --dry-run

    # Backfill a small subset
    python scripts/backfill_setup_engine.py --symbols AAPL,MSFT --yes

    # Backfill specific runs
    python scripts/backfill_setup_engine.py --run-id 42 --run-id 43 --yes

    # Backfill date range, skip confirmation
    python scripts/backfill_setup_engine.py --date-from 2025-01-01 --date-to 2025-06-30 --yes

    # Force overwrite existing setup_engine data
    python scripts/backfill_setup_engine.py --force --yes

Note: Best run during low-activity periods as it holds brief DB write locks.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — identical to other scripts in this directory
# ---------------------------------------------------------------------------
backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir))

import pandas as pd
from sqlalchemy import text

from app.analysis.patterns.models import SETUP_ENGINE_DEFAULT_SCHEMA_VERSION
from app.database import SessionLocal
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.scanners.base_screener import DataRequirements, StockData
from app.scanners.setup_engine_scanner import attach_setup_engine
from app.scanners.setup_engine_screener import SetupEngineScanner
from app.services.benchmark_cache_service import BenchmarkCacheService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Schema version we produce — rows already at this version are skipped.
CURRENT_SCHEMA = SETUP_ENGINE_DEFAULT_SCHEMA_VERSION

# Data requirements for the Setup Engine scanner
SETUP_ENGINE_REQUIREMENTS = DataRequirements(
    price_period="2y",
    needs_fundamentals=False,
    needs_quarterly_growth=False,
    needs_benchmark=True,
    needs_earnings_history=False,
)


# ───────────────────────────────────────────────────────────────────────────
# Row classification
# ───────────────────────────────────────────────────────────────────────────

def _needs_backfill(details_json: Any, force: bool) -> bool:
    """Determine if a feature row needs setup_engine backfill.

    Returns True when:
    - details_json is NULL / None
    - details_json has no "setup_engine" key
    - setup_engine.schema_version differs from current
    - ``force`` is True (always overwrite)
    """
    if force:
        return True
    if not details_json:
        return True
    if not isinstance(details_json, dict):
        return True
    se = details_json.get("setup_engine")
    if not se:
        return True
    if se.get("schema_version") != CURRENT_SCHEMA:
        return True
    return False


# ───────────────────────────────────────────────────────────────────────────
# Price data fetching
# ───────────────────────────────────────────────────────────────────────────

def _fetch_price_data(
    symbol: str,
    fetch_delay: float,
) -> Optional[pd.DataFrame]:
    """Fetch full price history for a symbol via PriceCacheService → yfinance."""
    try:
        from app.services.price_cache_service import PriceCacheService
        from app.services.yfinance_service import yfinance_service

        price_cache = PriceCacheService.get_instance()
        price_data = price_cache.get_historical_data(symbol, period="5y")

        if price_data is None or price_data.empty:
            logger.debug("Cache MISS for %s — fetching from yfinance", symbol)
            time.sleep(fetch_delay)
            price_data = yfinance_service.get_historical_data(
                symbol, period="5y", use_cache=False,
            )
        else:
            logger.debug(
                "Cache HIT for %s (%d days)", symbol, len(price_data),
            )

        if price_data is None or price_data.empty:
            return None
        return price_data
    except Exception as exc:
        logger.warning("Failed to fetch price data for %s: %s", symbol, exc)
        return None


def _truncate_to_date(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Return rows of *df* with index <= *as_of* (inclusive)."""
    if df.empty:
        return df
    cutoff = pd.Timestamp(as_of)
    return df.loc[df.index <= cutoff]


# ───────────────────────────────────────────────────────────────────────────
# Query helpers
# ───────────────────────────────────────────────────────────────────────────

def _build_query(
    db,
    run_ids: Optional[List[int]],
    date_from: Optional[date],
    date_to: Optional[date],
    symbols: Optional[List[str]],
    statuses: List[str],
):
    """Build the base query for target feature rows."""
    q = (
        db.query(
            StockFeatureDaily.run_id,
            StockFeatureDaily.symbol,
            StockFeatureDaily.as_of_date,
            StockFeatureDaily.details_json,
        )
        .join(FeatureRun, StockFeatureDaily.run_id == FeatureRun.id)
        .filter(FeatureRun.status.in_(statuses))
    )
    if run_ids:
        q = q.filter(StockFeatureDaily.run_id.in_(run_ids))
    if date_from:
        q = q.filter(FeatureRun.as_of_date >= date_from)
    if date_to:
        q = q.filter(FeatureRun.as_of_date <= date_to)
    if symbols:
        q = q.filter(StockFeatureDaily.symbol.in_(symbols))
    return q


# ───────────────────────────────────────────────────────────────────────────
# Spot-check verification
# ───────────────────────────────────────────────────────────────────────────

def _spot_check(
    db,
    updated_keys: List[Tuple[int, str]],
    sample_size: int = 5,
) -> bool:
    """Verify a few random backfilled rows have valid setup_engine data.

    Only samples from *updated_keys* (run_id, symbol pairs that were actually
    written during this backfill) so the check is meaningful even when only a
    subset of symbols was processed.
    """
    if not updated_keys:
        logger.info("Spot-check: no rows were updated — nothing to verify.")
        return True

    import random as _rng

    sample = _rng.sample(updated_keys, min(sample_size, len(updated_keys)))
    ok = 0
    for run_id, symbol in sample:
        row = (
            db.query(StockFeatureDaily.details_json)
            .filter(
                StockFeatureDaily.run_id == run_id,
                StockFeatureDaily.symbol == symbol,
            )
            .first()
        )
        if row is None:
            continue
        details = row[0]
        if not details or not isinstance(details, dict):
            continue
        se = details.get("setup_engine")
        if se and se.get("schema_version") == CURRENT_SCHEMA:
            ok += 1
    logger.info(
        "Spot-check: %d/%d sampled rows have valid setup_engine (schema=%s)",
        ok, len(sample), CURRENT_SCHEMA,
    )
    return ok == len(sample)


# ───────────────────────────────────────────────────────────────────────────
# Main backfill logic
# ───────────────────────────────────────────────────────────────────────────

def run_backfill(args: argparse.Namespace) -> None:
    """Execute the backfill pipeline."""
    db = SessionLocal()

    try:
        # SQLite busy timeout for concurrent access
        db.execute(text("PRAGMA busy_timeout = 30000"))

        # ── 1. Query target rows ──────────────────────────────────────
        statuses = [s.strip() for s in args.status.split(",")]
        all_rows = _build_query(
            db, args.run_id, args.date_from, args.date_to,
            args.symbols, statuses,
        ).all()

        total_queried = len(all_rows)
        logger.info("Queried %d feature rows (statuses=%s)", total_queried, statuses)

        if total_queried == 0:
            logger.info("No rows match the filter criteria. Nothing to do.")
            return

        # ── 2. Classify rows ──────────────────────────────────────────
        rows_needing_backfill: List[Tuple[int, str, date, Any]] = []
        already_ok = 0
        for run_id, symbol, as_of_date, details_json in all_rows:
            if _needs_backfill(details_json, args.force):
                rows_needing_backfill.append((run_id, symbol, as_of_date, details_json))
            else:
                already_ok += 1

        logger.info(
            "Summary: %d total, %d need backfill, %d already up-to-date",
            total_queried, len(rows_needing_backfill), already_ok,
        )

        if not rows_needing_backfill:
            logger.info("All rows already have setup_engine at schema %s. Done.", CURRENT_SCHEMA)
            return

        # ── 3. Dry-run exit ───────────────────────────────────────────
        if args.dry_run:
            # Show breakdown by symbol count
            sym_set = {sym for _, sym, _, _ in rows_needing_backfill}
            logger.info(
                "[DRY RUN] Would backfill %d rows across %d symbols. No writes performed.",
                len(rows_needing_backfill), len(sym_set),
            )
            return

        # ── 4. Confirmation ───────────────────────────────────────────
        if not args.yes:
            sym_set = {sym for _, sym, _, _ in rows_needing_backfill}
            answer = input(
                f"\nBackfill {len(rows_needing_backfill)} rows across "
                f"{len(sym_set)} symbols? [y/N] "
            )
            if answer.strip().lower() not in ("y", "yes"):
                logger.info("Aborted by user.")
                return

        # ── 5. Group by symbol ────────────────────────────────────────
        symbol_groups: Dict[str, List[Tuple[int, date, Any]]] = {}
        for run_id, symbol, as_of_date, details_json in rows_needing_backfill:
            symbol_groups.setdefault(symbol, []).append((run_id, as_of_date, details_json))

        symbols_list = sorted(symbol_groups.keys())
        total_symbols = len(symbols_list)
        total_rows = len(rows_needing_backfill)

        logger.info(
            "Processing %d rows across %d symbols (chunk_size=%d)",
            total_rows, total_symbols, args.chunk_size,
        )

        # ── 6. Fetch SPY benchmark data ONCE ──────────────────────────
        logger.info("Fetching SPY benchmark data...")
        benchmark_svc = BenchmarkCacheService.get_instance()
        spy_data = benchmark_svc.get_spy_data(period="2y")
        if spy_data is None or spy_data.empty:
            logger.error("Failed to fetch SPY benchmark data. Cannot proceed.")
            return
        logger.info("SPY benchmark: %d trading days loaded", len(spy_data))

        # ── 7. Process symbol chunks ──────────────────────────────────
        scanner = SetupEngineScanner()
        processed = 0
        skipped = 0
        errors = 0
        failed_symbols: List[str] = []
        updated_keys: List[Tuple[int, str]] = []  # (run_id, symbol) of written rows
        start_time = time.monotonic()

        for chunk_start in range(0, total_symbols, args.chunk_size):
            chunk_symbols = symbols_list[chunk_start:chunk_start + args.chunk_size]
            batch_updates: List[Tuple[dict, int, str]] = []  # (merged_details, run_id, symbol)

            for symbol in chunk_symbols:
                try:
                    # Fetch price data once per symbol
                    price_data = _fetch_price_data(symbol, args.fetch_delay)
                    if price_data is None or price_data.empty:
                        logger.warning("No price data for %s — skipping %d rows", symbol, len(symbol_groups[symbol]))
                        skipped += len(symbol_groups[symbol])
                        failed_symbols.append(symbol)
                        continue

                    # Process each (run_id, as_of_date) for this symbol
                    for run_id, as_of_date, existing_details in symbol_groups[symbol]:
                        try:
                            # Truncate price + SPY to as_of_date
                            truncated_prices = _truncate_to_date(price_data, as_of_date)
                            truncated_spy = _truncate_to_date(spy_data, as_of_date)

                            if truncated_prices.empty or len(truncated_prices) < 50:
                                logger.debug(
                                    "Insufficient data for %s @ %s (%d days) — skipping",
                                    symbol, as_of_date, len(truncated_prices),
                                )
                                skipped += 1
                                continue

                            # Build StockData and run the scanner
                            stock_data = StockData(
                                symbol=symbol,
                                price_data=truncated_prices,
                                benchmark_data=truncated_spy if not truncated_spy.empty else pd.DataFrame(),
                            )
                            result = scanner.scan_stock(symbol, stock_data)

                            # Extract the setup_engine payload from scanner result
                            se_payload = result.details.get("setup_engine")
                            if se_payload is None:
                                logger.info(
                                    "Scanner returned no setup_engine for %s @ %s (rating=%s) — skipping",
                                    symbol, as_of_date, result.rating,
                                )
                                skipped += 1
                                continue

                            # Merge into existing details_json
                            merged = attach_setup_engine(existing_details, se_payload)
                            batch_updates.append((merged, run_id, symbol))
                            updated_keys.append((run_id, symbol))
                            processed += 1

                        except Exception as row_exc:
                            logger.debug(
                                "Error processing %s run_id=%d @ %s: %s",
                                symbol, run_id, as_of_date, row_exc,
                            )
                            errors += 1

                except Exception as sym_exc:
                    logger.warning("Error processing symbol %s: %s", symbol, sym_exc)
                    errors += len(symbol_groups[symbol])
                    failed_symbols.append(symbol)

            # ── Batch UPDATE + commit checkpoint ──────────────────────
            if batch_updates:
                for merged_details, run_id, symbol in batch_updates:
                    db.query(StockFeatureDaily).filter(
                        StockFeatureDaily.run_id == run_id,
                        StockFeatureDaily.symbol == symbol,
                    ).update(
                        {"details_json": merged_details},
                        synchronize_session=False,
                    )
                db.commit()

            # ── Progress reporting ────────────────────────────────────
            chunk_end = min(chunk_start + args.chunk_size, total_symbols)
            elapsed = time.monotonic() - start_time
            symbols_done = chunk_end
            throughput = symbols_done / elapsed if elapsed > 0 else 0.0
            remaining = total_symbols - symbols_done
            eta = remaining / throughput if throughput > 0 else None
            eta_str = f"{eta:.0f}s" if eta is not None else "?"

            logger.info(
                "Progress: %d/%d symbols (%.1f%%) | "
                "rows: %d processed, %d skipped, %d errors | "
                "%.1f sym/s, ETA %s",
                symbols_done, total_symbols,
                100.0 * symbols_done / total_symbols,
                processed, skipped, errors,
                throughput, eta_str,
            )

        # ── 8. Post-backfill spot-check ───────────────────────────────
        logger.info("Running post-backfill spot-check...")
        _spot_check(db, updated_keys)

        # ── 9. Final summary ──────────────────────────────────────────
        elapsed = time.monotonic() - start_time
        logger.info("=" * 60)
        logger.info("Backfill complete in %.1fs", elapsed)
        logger.info("  Processed: %d rows", processed)
        logger.info("  Skipped:   %d rows", skipped)
        logger.info("  Errors:    %d rows", errors)
        if failed_symbols:
            logger.info(
                "  Failed symbols (%d): %s",
                len(failed_symbols),
                ", ".join(sorted(set(failed_symbols))[:20]),
            )
            if len(set(failed_symbols)) > 20:
                logger.info("    ... and %d more", len(set(failed_symbols)) - 20)
        logger.info("=" * 60)

    except Exception:
        logger.exception("Fatal error during backfill")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def _parse_date(value: str) -> date:
    """Parse YYYY-MM-DD date string."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date: {value!r} (expected YYYY-MM-DD)")


def _parse_symbols(value: str) -> List[str]:
    """Parse comma-separated symbol list, uppercased."""
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Backfill setup_engine payloads into historical feature store rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-id", type=int, action="append", default=None,
        help="Target specific feature run ID(s). Repeatable.",
    )
    parser.add_argument(
        "--date-from", type=_parse_date, default=None,
        help="Only runs with as_of_date >= this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date-to", type=_parse_date, default=None,
        help="Only runs with as_of_date <= this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--symbols", type=_parse_symbols, default=None,
        help="Comma-separated symbols to backfill (e.g. AAPL,MSFT,GOOG).",
    )
    parser.add_argument(
        "--status", type=str, default="published,completed",
        help="Comma-separated run statuses to target (default: published,completed).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Preview counts without writing to the database.",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite existing setup_engine even if schema_version matches.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="Number of symbols to process per batch (default: 50).",
    )
    parser.add_argument(
        "--fetch-delay", type=float, default=0.5,
        help="Seconds to sleep between yfinance fetches on cache miss (default: 0.5).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", default=False,
        help="Skip confirmation prompt.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_backfill(args)


if __name__ == "__main__":
    main()
