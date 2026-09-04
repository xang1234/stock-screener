"""Read-only queries for already published options analytics."""

from __future__ import annotations

from typing import Any


class OptionsAnalyticsQueries:
    def __init__(self, repository: Any, *, calculation_version: str) -> None:
        self._repository = repository
        self._calculation_version = calculation_version

    def get_published_command_center(self, market: str) -> Any:
        return self._repository.get_published_run(
            market.strip().upper(), self._calculation_version
        )

    def get_published_symbol_detail(self, symbol: str, market: str) -> Any:
        return self._repository.get_published_symbol_detail(
            symbol.strip().upper(),
            market.strip().upper(),
            self._calculation_version,
        )

    def get_run_diagnostics(self, run_id: int) -> Any:
        return self._repository.get_run_diagnostics(run_id)

