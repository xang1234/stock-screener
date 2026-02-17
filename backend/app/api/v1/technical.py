"""
Technical analysis API endpoints.
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging
import numpy as np

from ...wiring.bootstrap import get_scan_orchestrator
from ...services.yfinance_service import yfinance_service

logger = logging.getLogger(__name__)
router = APIRouter()


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        # Handle NaN and Inf values
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif obj is None:
        return None
    elif isinstance(obj, float):
        # Handle regular Python floats that might be NaN or Inf
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj


@router.get("/{symbol}/minervini")
async def scan_minervini(
    symbol: str,
    include_vcp: bool = Query(True, description="Include VCP pattern detection"),
):
    """
    Scan stock using Minervini template criteria.

    Analyzes:
    - Relative Strength (RS) rating
    - Weinstein Stage (1-4)
    - Moving Average alignment
    - 52-week positioning
    - VCP pattern (optional)

    Returns comprehensive Minervini analysis with pass/fail and score.
    """
    try:
        logger.info(f"Minervini scan request for {symbol}, include_vcp={include_vcp}")
        orchestrator = get_scan_orchestrator()
        result = orchestrator.scan_stock_multi(
            symbol=symbol.upper(),
            screener_names=["minervini"],
            criteria={"include_vcp": include_vcp}
        )
        logger.info(f"Minervini scan completed for {symbol}, score={result.get('minervini_score')}")
        # Convert numpy types to native Python types for JSON serialization
        return convert_numpy_types(result)
    except Exception as e:
        logger.error(f"Error in Minervini scan for {symbol}: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error scanning {symbol}: {str(e)}")


@router.get("/{symbol}/rs-rating")
async def get_rs_rating(symbol: str):
    """
    Get Relative Strength (RS) rating for a stock.

    Calculates weighted performance vs SPY benchmark over multiple periods:
    - 63 days (Q1): 40% weight
    - 126 days (Q2): 20% weight
    - 189 days (Q3): 20% weight
    - 252 days (1 year): 20% weight

    Returns RS rating (0-100) and performance breakdown.
    """
    from ...scanners.criteria.relative_strength import RelativeStrengthCalculator

    # Fetch price data (2 years to ensure we have 252+ trading days)
    stock_data = yfinance_service.get_historical_data(symbol.upper(), period="2y")
    spy_data = yfinance_service.get_historical_data("SPY", period="2y")

    if stock_data is None or spy_data is None:
        return {"error": "Unable to fetch price data"}

    stock_prices = stock_data["Close"][::-1].reset_index(drop=True)
    spy_prices = spy_data["Close"][::-1].reset_index(drop=True)

    calc = RelativeStrengthCalculator()
    result = calc.calculate_rs_rating(symbol.upper(), stock_prices, spy_prices)

    # Add period returns
    period_returns = calc.calculate_period_returns(stock_prices)

    response = {
        "symbol": symbol.upper(),
        "rs_rating": result["rs_rating"],
        "relative_performance": result["relative_performance"],
        "period_returns": period_returns,
        "benchmark": "SPY",
    }
    return convert_numpy_types(response)


@router.get("/{symbol}/stage")
async def get_stage_analysis(symbol: str):
    """
    Get Weinstein Stage Analysis for a stock.

    Stages:
    - Stage 1: Basing (sideways below declining MA)
    - Stage 2: Advancing (uptrend above rising MA) - IDEAL
    - Stage 3: Topping (sideways above flattening MA)
    - Stage 4: Declining (downtrend below declining MA)

    Returns stage number, trend information, and confidence score.
    """
    from ...scanners.criteria.stage_analysis import WeinsteinstageAnalyzer

    # Fetch price data (2 years to ensure we have 252+ trading days)
    data = yfinance_service.get_historical_data(symbol.upper(), period="2y")

    if data is None:
        return {"error": "Unable to fetch price data"}

    # Keep chronological order for rolling calculations, then reverse
    prices_chrono = data["Close"].reset_index(drop=True)
    volumes_chrono = data["Volume"].reset_index(drop=True)

    # Calculate 200-day MA on chronological data
    ma_200_series_chrono = prices_chrono.rolling(window=200, min_periods=200).mean()

    # Get most recent values (last in chronological series)
    current_price = prices_chrono.iloc[-1]
    ma_200 = ma_200_series_chrono.iloc[-1]

    # Reverse for passing to analyzer (most recent first)
    prices = prices_chrono[::-1].reset_index(drop=True)
    volumes = volumes_chrono[::-1].reset_index(drop=True)
    ma_200_series = ma_200_series_chrono[::-1].reset_index(drop=True)

    analyzer = WeinsteinstageAnalyzer()
    result = analyzer.determine_stage(
        current_price,
        ma_200,
        ma_200_series,
        prices,
        volumes
    )

    response = {
        "symbol": symbol.upper(),
        **result
    }
    return convert_numpy_types(response)


@router.get("/{symbol}/ma-analysis")
async def get_ma_analysis(symbol: str):
    """
    Get Moving Average alignment analysis.

    Checks Minervini template criteria:
    - Price > 50-day > 150-day > 200-day MA
    - 200-day MA trending up for 1+ month
    - 50-day MA above both 150 and 200-day MA

    Returns alignment status, scores, and detailed breakdown.
    """
    from ...scanners.criteria.moving_averages import MovingAverageAnalyzer

    # Fetch price data (2 years to ensure we have 252+ trading days)
    data = yfinance_service.get_historical_data(symbol.upper(), period="2y")

    if data is None:
        return {"error": "Unable to fetch price data"}

    # Keep chronological order for rolling calculations
    prices_chrono = data["Close"].reset_index(drop=True)

    # Calculate MAs on chronological data
    current_price = prices_chrono.iloc[-1]
    ma_50 = prices_chrono.rolling(window=50, min_periods=50).mean().iloc[-1]
    ma_150 = prices_chrono.rolling(window=150, min_periods=150).mean().iloc[-1]
    ma_200 = prices_chrono.rolling(window=200, min_periods=200).mean().iloc[-1]
    ma_200_month_ago = prices_chrono.rolling(window=200, min_periods=200).mean().iloc[-21] if len(prices_chrono) > 220 else ma_200

    analyzer = MovingAverageAnalyzer()
    result = analyzer.comprehensive_ma_analysis(
        current_price,
        ma_50,
        ma_150,
        ma_200,
        ma_200_month_ago
    )

    response = {
        "symbol": symbol.upper(),
        "current_price": current_price,
        "ma_50": ma_50,
        "ma_150": ma_150,
        "ma_200": ma_200,
        **result
    }
    return convert_numpy_types(response)


@router.get("/{symbol}/vcp")
async def detect_vcp(symbol: str):
    """
    Detect Volatility Contraction Pattern (VCP).

    VCP characteristics:
    - 3-4 consolidation bases with contracting ranges
    - Each pullback shallower than previous
    - Volume decreases on pullbacks
    - Price tight near recent highs

    Returns VCP detection status, score, and pattern details.
    """
    from ...scanners.criteria.vcp_detection import VCPDetector

    # Fetch price data
    data = yfinance_service.get_historical_data(symbol.upper(), period="6mo")

    if data is None:
        return {"error": "Unable to fetch price data"}

    prices = data["Close"][::-1].reset_index(drop=True)
    volumes = data["Volume"][::-1].reset_index(drop=True)

    detector = VCPDetector()
    result = detector.detect_vcp(prices, volumes)

    response = {
        "symbol": symbol.upper(),
        **result
    }
    return convert_numpy_types(response)


@router.get("/{symbol}/52-week-position")
async def get_52w_position(symbol: str):
    """
    Get stock position relative to 52-week range.

    Minervini criteria:
    - At least 30% above 52-week low
    - Within 25% of 52-week high

    Returns positioning metrics and whether criteria are met.
    """
    # Fetch price data (2 years to ensure we have 252+ trading days)
    data = yfinance_service.get_historical_data(symbol.upper(), period="2y")

    if data is None:
        return {"error": "Unable to fetch price data"}

    prices = data["Close"][::-1].reset_index(drop=True)
    current_price = prices.iloc[0]
    high_52w = prices.max()
    low_52w = prices.min()

    above_low_pct = ((current_price - low_52w) / low_52w) * 100 if low_52w > 0 else None
    from_high_pct = ((high_52w - current_price) / high_52w) * 100 if high_52w > 0 else None

    response = {
        "symbol": symbol.upper(),
        "current_price": current_price,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "above_low_pct": round(above_low_pct, 2) if above_low_pct else None,
        "from_high_pct": round(from_high_pct, 2) if from_high_pct else None,
        "meets_low_criteria": above_low_pct >= 30 if above_low_pct else False,
        "meets_high_criteria": from_high_pct <= 25 if from_high_pct else False,
        "ideal_position": (
            above_low_pct >= 30 and from_high_pct <= 10
        ) if (above_low_pct and from_high_pct) else False,
    }
    return convert_numpy_types(response)
