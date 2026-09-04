from __future__ import annotations

from app.use_cases.options_analytics.queries import OptionsAnalyticsQueries
from app.wiring import bootstrap


class _Repository:
    def get_published_run(self, market, calculation_version):
        return (market, calculation_version, "published")

    def get_published_symbol_detail(self, symbol, market, calculation_version):
        return (symbol, market, calculation_version)

    def get_run_diagnostics(self, run_id):
        return {"run_id": run_id}


def test_queries_are_read_only_repository_delegates() -> None:
    queries = OptionsAnalyticsQueries(_Repository(), calculation_version="v1")

    assert queries.get_published_command_center("us") == ("US", "v1", "published")
    assert queries.get_published_symbol_detail("aapl", "us") == ("AAPL", "US", "v1")
    assert queries.get_run_diagnostics(7) == {"run_id": 7}


def test_options_use_cases_are_exposed_through_application_wiring() -> None:
    assert callable(bootstrap.get_refresh_options_analytics_use_case)
    assert callable(bootstrap.get_options_analytics_queries)
