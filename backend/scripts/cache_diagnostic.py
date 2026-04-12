"""
Diagnostic script to trace the bulk refresh → cache → database flow.

This will help identify where the 19-day limitation is coming from.
"""
import logging
from app.models.stock import StockPrice
from app.wiring.bootstrap import (
    get_cache_manager,
    get_price_cache,
    get_session_factory,
    initialize_process_runtime_services,
)
from sqlalchemy import func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test symbols
test_symbols = ['AAPL', 'MSFT', 'GOOGL']

print("=" * 80)
print("DIAGNOSTIC: Tracing Bulk Refresh → Cache → Database Flow")
print("=" * 80)

initialize_process_runtime_services()

# Step 1: Check what's currently in the database
print("\n1. CURRENT DATABASE STATE")
print("-" * 80)
db = get_session_factory()()
try:
    for symbol in test_symbols:
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
            print(f"{symbol}: {count} rows (from {min_date} to {max_date})")
        else:
            print(f"{symbol}: NO DATA in database")
finally:
    db.close()

# Step 2: Run a bulk refresh
print("\n2. RUNNING BULK REFRESH (FORCE=TRUE)")
print("-" * 80)
cache_manager = get_cache_manager()

# Force refresh to ensure we bypass any cache
result = cache_manager.warm_price_cache(
    test_symbols,
    period="2y",
    batch_size=3,
    rate_limit=0.5,
    force_refresh=True  # Force full refresh
)

print(f"Bulk refresh result: {result['successful']} successful, {result['failed']} failed")

# Step 3: Check database again
print("\n3. DATABASE STATE AFTER REFRESH")
print("-" * 80)
db = get_session_factory()()
try:
    for symbol in test_symbols:
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
            print(f"{symbol}: {count} rows (from {min_date} to {max_date})")
        else:
            print(f"{symbol}: NO DATA in database")
finally:
    db.close()

# Step 4: Try to retrieve from cache
print("\n4. RETRIEVING FROM CACHE (period=2y)")
print("-" * 80)
price_cache = get_price_cache()

for symbol in test_symbols:
    data = price_cache.get_historical_data(symbol, period="2y", force_refresh=False)

    if data is not None and not data.empty:
        print(f"{symbol}: {len(data)} days (from {data.index[0].date()} to {data.index[-1].date()})")
    else:
        print(f"{symbol}: NO DATA from cache")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("If database shows ~502 rows but cache returns only ~19 days,")
print("then there's an issue in the _get_from_database() method.")
print()
print("If database shows only ~19 rows, then bulk_fetcher isn't")
print("storing the full data, or there's a date filter issue.")
print("=" * 80)
