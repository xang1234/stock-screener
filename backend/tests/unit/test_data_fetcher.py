from app.services.alphavantage_service import AlphaVantageService
from app.services.data_fetcher import DataFetcher
from app.services.yfinance_service import YFinanceService
from app.wiring.bootstrap import clear_runtime_services


def test_data_fetcher_supports_explicit_dependency_injection():
    yfinance_service = object()
    alphavantage_service = object()

    fetcher = DataFetcher(
        yfinance_service=yfinance_service,
        alphavantage_service=alphavantage_service,
    )

    assert fetcher._yfinance_service is yfinance_service
    assert fetcher._alphavantage_service is alphavantage_service


def test_data_fetcher_constructs_without_runtime_container():
    clear_runtime_services()

    fetcher = DataFetcher()

    assert isinstance(fetcher._yfinance_service, YFinanceService)
    assert isinstance(fetcher._alphavantage_service, AlphaVantageService)
