"""
Test script for manual breadth calculation.

This script tests the breadth calculation service directly
without going through the Celery task queue.
"""
import sys
from pathlib import Path
from datetime import datetime

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.database import SessionLocal
from app.services.breadth_calculator_service import BreadthCalculatorService
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.wiring.bootstrap import get_price_cache

def test_breadth_calculation():
    """Run a test breadth calculation."""
    print("=" * 60)
    print("Testing Breadth Calculation Service")
    print("=" * 60)

    db = SessionLocal()

    try:
        # Check how many active stocks we have
        active_count = db.query(StockUniverse).filter(
            StockUniverse.is_active == True
        ).count()

        print(f"\n📊 Active stocks in universe: {active_count}")

        if active_count == 0:
            print("\n⚠️  No active stocks found in StockUniverse!")
            print("   Please populate the universe table first.")
            return

        # Initialize service
        calculator = BreadthCalculatorService(db, get_price_cache())

        # Calculate breadth for today
        print(f"\n🔄 Calculating breadth for today ({datetime.now().date()})...")
        print("   This may take 30-60 seconds for large universe...")

        metrics = calculator.calculate_daily_breadth()

        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\n📈 Daily Movers (4%+ threshold):")
        print(f"   Stocks up 4%+:    {metrics['stocks_up_4pct']}")
        print(f"   Stocks down 4%+:  {metrics['stocks_down_4pct']}")

        print(f"\n📊 Multi-Day Ratios:")
        print(f"   5-day ratio:      {metrics['ratio_5day']}")
        print(f"   10-day ratio:     {metrics['ratio_10day']}")

        print(f"\n📅 Monthly Movers (21 days, 25%+ threshold):")
        print(f"   Stocks up 25%+:   {metrics['stocks_up_25pct_month']}")
        print(f"   Stocks down 25%+: {metrics['stocks_down_25pct_month']}")

        print(f"\n📅 Monthly Movers (21 days, 50%+ threshold):")
        print(f"   Stocks up 50%+:   {metrics['stocks_up_50pct_month']}")
        print(f"   Stocks down 50%+: {metrics['stocks_down_50pct_month']}")

        print(f"\n📅 34-Day Movers (13%+ threshold):")
        print(f"   Stocks up 13%+:   {metrics['stocks_up_13pct_34days']}")
        print(f"   Stocks down 13%+: {metrics['stocks_down_13pct_34days']}")

        print(f"\n📅 Quarterly Movers (63 days, 25%+ threshold):")
        print(f"   Stocks up 25%+:   {metrics['stocks_up_25pct_quarter']}")
        print(f"   Stocks down 25%+: {metrics['stocks_down_25pct_quarter']}")

        print(f"\n📊 Metadata:")
        print(f"   Total stocks scanned: {metrics['total_stocks_scanned']}")
        print(f"   Skipped stocks:       {metrics.get('skipped_stocks', 0)}")

        # Save to database
        print("\n💾 Saving to database...")
        breadth_record = MarketBreadth(
            date=datetime.now().date(),
            stocks_up_4pct=metrics['stocks_up_4pct'],
            stocks_down_4pct=metrics['stocks_down_4pct'],
            ratio_5day=metrics['ratio_5day'],
            ratio_10day=metrics['ratio_10day'],
            stocks_up_25pct_quarter=metrics['stocks_up_25pct_quarter'],
            stocks_down_25pct_quarter=metrics['stocks_down_25pct_quarter'],
            stocks_up_25pct_month=metrics['stocks_up_25pct_month'],
            stocks_down_25pct_month=metrics['stocks_down_25pct_month'],
            stocks_up_50pct_month=metrics['stocks_up_50pct_month'],
            stocks_down_50pct_month=metrics['stocks_down_50pct_month'],
            stocks_up_13pct_34days=metrics['stocks_up_13pct_34days'],
            stocks_down_13pct_34days=metrics['stocks_down_13pct_34days'],
            total_stocks_scanned=metrics['total_stocks_scanned']
        )

        db.merge(breadth_record)
        db.commit()

        print("✓ Breadth data saved successfully!")

        # Verify save
        saved_record = db.query(MarketBreadth).filter(
            MarketBreadth.date == datetime.now().date()
        ).first()

        if saved_record:
            print(f"✓ Verified: Record exists in database (ID: {saved_record.id})")

        print("\n" + "=" * 60)
        print("✓ Test Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()

    finally:
        db.close()

if __name__ == "__main__":
    test_breadth_calculation()
