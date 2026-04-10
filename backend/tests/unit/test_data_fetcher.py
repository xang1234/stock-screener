from app.services.alphavantage_service import AlphaVantageService
from app.services.data_fetcher import DataFetcher
from app.services.yfinance_service import YFinanceService


def test_data_fetcher_supports_explicit_dependency_injection():
    yfinance_service = object()
    alphavantage_service = object()

    fetcher = DataFetcher(
        yfinance_service=yfinance_service,
        alphavantage_service=alphavantage_service,
    )

    assert fetcher._yfinance_service is yfinance_service
    assert fetcher._alphavantage_service is alphavantage_service


def test_data_fetcher_constructs_without_runtime_container(monkeypatch):
    expected_yfinance = YFinanceService()
    expected_alphavantage = AlphaVantageService()
    monkeypatch.setattr(
        DataFetcher,
        "_build_default_yfinance_service",
        staticmethod(lambda: expected_yfinance),
    )
    monkeypatch.setattr(
        DataFetcher,
        "_build_default_alphavantage_service",
        staticmethod(lambda: expected_alphavantage),
    )
    fetcher = DataFetcher()

    assert fetcher._yfinance_service is expected_yfinance
    assert fetcher._alphavantage_service is expected_alphavantage
