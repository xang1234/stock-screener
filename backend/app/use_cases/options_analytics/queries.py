"""Read-only queries for already published options analytics."""

from __future__ import annotations

from dataclasses import dataclass
from app.use_cases.options_analytics.ports import (
    OptionsRunItemRecord,
    OptionsRunRecord,
    PublishedOptionsReader,
)


@dataclass(frozen=True)
class PublishedOptionsSymbolDetail:
    run: OptionsRunRecord
    item: OptionsRunItemRecord
    history: tuple[OptionsRunItemRecord, ...]


class OptionsAnalyticsQueries:
    def __init__(
        self,
        repository: PublishedOptionsReader,
        *,
        calculation_version: str,
    ) -> None:
        self._repository = repository
        self._calculation_version = calculation_version

    def get_published_command_center(self, market: str) -> OptionsRunRecord | None:
        return self._repository.get_published_run(
            market.strip().upper(), self._calculation_version
        )

    def get_published_symbol_detail(
        self,
        symbol: str,
        market: str,
    ) -> PublishedOptionsSymbolDetail | None:
        canonical_symbol = symbol.strip().upper()
        canonical_market = market.strip().upper()
        item = self._repository.get_published_symbol_detail(
            canonical_symbol,
            canonical_market,
            self._calculation_version,
        )
        if item is None:
            return None
        return PublishedOptionsSymbolDetail(
            run=item.run,
            item=item,
            history=self._repository.symbol_history(
                canonical_symbol,
                market=canonical_market,
                calculation_version=self._calculation_version,
            ),
        )

    def get_run_diagnostics(self, run_id: int) -> OptionsRunRecord | None:
        return self._repository.get_run_diagnostics(run_id)

    def is_stale(self, run: OptionsRunRecord, market: str) -> bool:
        latest_source_run_id = self._repository.latest_source_feature_run_id(
            market.strip().upper()
        )
        return (
            latest_source_run_id is not None
            and latest_source_run_id != run.source_feature_run_id
        )
