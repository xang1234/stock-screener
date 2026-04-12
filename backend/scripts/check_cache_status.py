"""
Check the current status of price cache (database + Redis).

This will help diagnose the "19 days" issue by showing exactly
what data is stored where.
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import logging
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.wiring.bootstrap import (
    get_price_cache,
    get_session_factory,
    initialize_process_runtime_services,
)
from sqlalchemy import func

logging.basicConfig(level=logging.WARNING)  # Quiet output

print("=" * 80)
print("CACHE STATUS DIAGNOSTIC")
print("=" * 80)

initialize_process_runtime_services()
db = get_session_factory()()
price_cache = get_price_cache()

try:
    # Sample 10 active stocks
    universe_stocks = db.query(StockUniverse).filter(
        StockUniverse.is_active == True
    ).limit(10).all()

    symbols = [s.symbol for s in universe_stocks]

    print(f"\nChecking {len(symbols)} sample symbols: {', '.join(symbols)}")
    print("\n" + "-" * 80)
    print(f"{'Symbol':<10} {'DB Rows':<10} {'DB Range':<40} {'Cache Days':<15}")
    print("-" * 80)

    issues = []

    for symbol in symbols:
        # Check database
        count = db.query(func.count(StockPrice.id)).filter(
            StockPrice.symbol == symbol
        ).scalar()

        if count > 0:
            min_date = db.query(func.min(StockPrice.date)).filter(
                StockPrice.symbol == symbol
            ).scalar()
            max_date = db.query(func.max(StockPrice.date)).filter(
                StockPrice.symbol == symbol
            ).scalar()
            db_range = f"{min_date} to {max_date}"
        else:
            db_range = "NO DATA"

        # Check cache retrieval
        cached_data = price_cache.get_historical_data(symbol, period="2y", force_refresh=False)
        cache_days = len(cached_data) if cached_data is not None else 0

        print(f"{symbol:<10} {count:<10} {db_range:<40} {cache_days:<15}")

        # Flag issues
        if count < 400:  # Less than ~1.5 years
            issues.append(f"{symbol}: Only {count} rows in database (expected ~500)")
        if cache_days < 400:
            issues.append(f"{symbol}: Only {cache_days} days in cache (expected ~500)")

    print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")

        print("\nRECOMMENDATION:")
        print("  Run: python scripts/force_full_cache_refresh.py")
        print("  This will clear and repopulate cache with full 2-year data.")
    else:
        print("\n✓ All sampled stocks have sufficient data (~500 days)")
        print("  Cache is properly populated with 2 years of price data.")

    print("=" * 80)

finally:
    db.close()
