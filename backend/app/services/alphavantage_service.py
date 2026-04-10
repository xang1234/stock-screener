"""
Alpha Vantage service wrapper for fetching fundamental data.
"""
import requests
from typing import Optional, Dict, Any
import logging
from ..config import settings

logger = logging.getLogger(__name__)


class AlphaVantageService:
    """Service for fetching fundamental data using Alpha Vantage API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage service.

        Args:
            api_key: Alpha Vantage API key (uses config if not provided)
        """
        self.api_key = api_key or settings.alpha_vantage_api_key

        if not self.api_key:
            logger.warning("Alpha Vantage API key not configured")

    def _make_request(self, function: str, symbol: str, **kwargs) -> Optional[Dict]:
        """
        Make API request to Alpha Vantage.

        Args:
            function: API function name
            symbol: Stock ticker symbol
            **kwargs: Additional query parameters

        Returns:
            JSON response or None if error
        """
        if not self.api_key:
            logger.error("Cannot make request without API key")
            return None

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            **kwargs
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None

            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return None

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {e}")
            return None
        except ValueError as e:
            logger.error(f"JSON parse error for {symbol}: {e}")
            return None

    def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company overview with fundamental data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with company data or None if error
        """
        data = self._make_request("OVERVIEW", symbol)

        if not data or not data.get("Symbol"):
            return None

        try:
            return {
                "symbol": data.get("Symbol"),
                "name": data.get("Name"),
                "sector": data.get("Sector"),
                "industry": data.get("Industry"),
                "market_cap": int(data.get("MarketCapitalization", 0)) if data.get("MarketCapitalization") else None,
                "pe_ratio": float(data.get("PERatio", 0)) if data.get("PERatio") != "None" else None,
                "peg_ratio": float(data.get("PEGRatio", 0)) if data.get("PEGRatio") != "None" else None,
                "price_to_book": float(data.get("PriceToBookRatio", 0)) if data.get("PriceToBookRatio") != "None" else None,
                "eps": float(data.get("EPS", 0)) if data.get("EPS") != "None" else None,
                "profit_margin": float(data.get("ProfitMargin", 0)) if data.get("ProfitMargin") != "None" else None,
                "roe": float(data.get("ReturnOnEquityTTM", 0)) if data.get("ReturnOnEquityTTM") != "None" else None,
                "revenue_ttm": int(data.get("RevenueTTM", 0)) if data.get("RevenueTTM") else None,
                "shares_outstanding": int(data.get("SharesOutstanding", 0)) if data.get("SharesOutstanding") else None,
            }

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing overview data for {symbol}: {e}")
            return None

    def get_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get earnings data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with earnings data or None if error
        """
        data = self._make_request("EARNINGS", symbol)

        if not data:
            return None

        try:
            quarterly = data.get("quarterlyEarnings", [])
            annual = data.get("annualEarnings", [])

            if not quarterly:
                return None

            # Get most recent quarterly earnings
            recent_quarter = quarterly[0] if len(quarterly) > 0 else {}
            year_ago_quarter = quarterly[4] if len(quarterly) > 4 else {}

            # Calculate quarterly EPS growth
            eps_current = float(recent_quarter.get("reportedEPS", 0))
            eps_year_ago = float(year_ago_quarter.get("reportedEPS", 0)) if year_ago_quarter else 0

            eps_growth_quarterly = None
            if eps_year_ago and eps_year_ago > 0:
                eps_growth_quarterly = ((eps_current - eps_year_ago) / abs(eps_year_ago)) * 100

            # Calculate annual EPS growth
            eps_growth_annual = None
            if len(annual) >= 3:
                recent_years = annual[:3]
                eps_values = [float(y.get("reportedEPS", 0)) for y in recent_years]

                # Calculate average annual growth
                if all(eps > 0 for eps in eps_values[1:]):
                    growth_rates = []
                    for i in range(len(eps_values) - 1):
                        if eps_values[i + 1] > 0:
                            growth = ((eps_values[i] - eps_values[i + 1]) / eps_values[i + 1]) * 100
                            growth_rates.append(growth)

                    if growth_rates:
                        eps_growth_annual = sum(growth_rates) / len(growth_rates)

            return {
                "symbol": symbol,
                "eps_current": eps_current,
                "eps_growth_quarterly": eps_growth_quarterly,
                "eps_growth_annual": eps_growth_annual,
                "quarterly_earnings": quarterly[:8],  # Last 2 years
                "annual_earnings": annual[:5],  # Last 5 years
            }

        except (ValueError, TypeError, KeyError, IndexError) as e:
            logger.error(f"Error parsing earnings data for {symbol}: {e}")
            return None

    def get_income_statement(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get income statement data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with income statement data or None if error
        """
        data = self._make_request("INCOME_STATEMENT", symbol)

        if not data:
            return None

        try:
            quarterly = data.get("quarterlyReports", [])
            annual = data.get("annualReports", [])

            if not quarterly:
                return None

            # Get most recent data
            recent = quarterly[0]
            year_ago = quarterly[4] if len(quarterly) > 4 else {}

            revenue_current = int(recent.get("totalRevenue", 0))
            revenue_year_ago = int(year_ago.get("totalRevenue", 0)) if year_ago else 0

            revenue_growth = None
            if revenue_year_ago and revenue_year_ago > 0:
                revenue_growth = ((revenue_current - revenue_year_ago) / revenue_year_ago) * 100

            return {
                "symbol": symbol,
                "revenue_current": revenue_current,
                "revenue_growth": revenue_growth,
                "profit_margin": float(recent.get("profitMargin", 0)) if recent.get("profitMargin") else None,
                "operating_margin": float(recent.get("operatingMargin", 0)) if recent.get("operatingMargin") else None,
            }

        except (ValueError, TypeError, KeyError, IndexError) as e:
            logger.error(f"Error parsing income statement for {symbol}: {e}")
            return None
