#!/usr/bin/env python3
"""
Test script for MinerviniScanner and multi-screener architecture.

Tests:
1. Screener registry
2. Data preparation layer
3. MinerviniScanner implementation
4. Scan orchestrator
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.scanners.screener_registry import screener_registry
from app.wiring.bootstrap import get_scan_orchestrator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test symbols
TEST_SYMBOLS = ['AAPL', 'MSFT', 'NVDA']


def test_screener_registry():
    """Test that MinerviniScanner is registered."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Screener Registry")
    logger.info("="*80)

    # Check if minervini screener is registered
    assert screener_registry.is_registered('minervini'), "Minervini screener not registered!"
    logger.info("✓ Minervini screener is registered")

    # Get screener instance
    screener = screener_registry.get('minervini')
    assert screener is not None, "Failed to get screener instance"
    logger.info(f"✓ Got screener instance: {screener}")

    # Check screener name
    assert screener.screener_name == 'minervini', "Incorrect screener name"
    logger.info(f"✓ Screener name: {screener.screener_name}")

    # List all screeners
    all_screeners = screener_registry.list_screeners()
    logger.info(f"✓ Registered screeners: {all_screeners}")

    logger.info("\n✅ Registry test passed!\n")


def test_data_requirements():
    """Test data requirements from screener."""
    logger.info("="*80)
    logger.info("TEST 2: Data Requirements")
    logger.info("="*80)

    screener = screener_registry.get('minervini')
    requirements = screener.get_data_requirements()

    logger.info(f"Price period: {requirements.price_period}")
    logger.info(f"Needs fundamentals: {requirements.needs_fundamentals}")
    logger.info(f"Needs quarterly growth: {requirements.needs_quarterly_growth}")
    logger.info(f"Needs benchmark: {requirements.needs_benchmark}")
    logger.info(f"Needs earnings history: {requirements.needs_earnings_history}")

    assert requirements.price_period == "2y", "Incorrect price period"
    assert requirements.needs_benchmark == True, "Should need benchmark"
    # Minervini doesn't use quarterly growth in scoring (only informational)
    assert requirements.needs_quarterly_growth == False, "Minervini doesn't need quarterly growth"

    logger.info("\n✅ Data requirements test passed!\n")


def test_scan_orchestrator():
    """Test the scan orchestrator with MinerviniScanner."""
    logger.info("="*80)
    logger.info("TEST 3: Scan Orchestrator")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    for symbol in TEST_SYMBOLS:
        logger.info(f"\n📊 Scanning {symbol}...")
        logger.info("-" * 80)

        try:
            result = orchestrator.scan_stock_multi(
                symbol=symbol,
                screener_names=['minervini'],
                criteria={'include_vcp': False}  # Skip VCP for speed
            )

            # Check result structure
            assert 'symbol' in result, "Missing symbol in result"
            assert 'composite_score' in result, "Missing composite_score in result"
            assert 'rating' in result, "Missing rating in result"
            assert 'minervini_score' in result, "Missing minervini_score in result"

            logger.info(f"Symbol: {result['symbol']}")
            logger.info(f"Composite Score: {result['composite_score']:.2f}")
            _ms = result.get('minervini_score')
            logger.info(f"Minervini Score: {_ms:.2f}" if _ms is not None else "Minervini Score: N/A")
            logger.info(f"Rating: {result['rating']}")
            logger.info(f"Current Price: ${result.get('current_price', 'N/A')}")
            logger.info(f"RS Rating: {result.get('rs_rating', 'N/A')}")
            logger.info(f"Stage: {result.get('stage', 'N/A')}")
            logger.info(f"Screeners Run: {result.get('screeners_run', [])}")

            if 'error' in result:
                logger.warning(f"⚠️  Error: {result['error']}")
            else:
                logger.info("✓ Scan completed successfully")

        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            raise

    logger.info("\n✅ Orchestrator test passed!\n")


def main():
    """Run all tests."""
    logger.info("\n" + "🚀 " + "="*76)
    logger.info("MINERVINI SCANNER V2 TEST SUITE")
    logger.info("="*78)

    try:
        test_screener_registry()
        test_data_requirements()
        test_scan_orchestrator()

        logger.info("\n" + "🎉 " + "="*76)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*78 + "\n")

        logger.info("Summary:")
        logger.info("  ✅ Screener registry working")
        logger.info("  ✅ Data requirements correct")
        logger.info("  ✅ Scan orchestrator functional")
        logger.info("\nMinervini scanner V2 via orchestrator is ready!")

        return True

    except Exception as e:
        logger.error(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
