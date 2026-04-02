"""Adapter wrapping DataPreparationLayer as a StockDataProvider port."""

from __future__ import annotations

from app.domain.scanning.ports import StockDataProvider
from app.scanners.data_preparation import DataPreparationLayer


class DataPrepStockDataProvider(StockDataProvider):
    """Delegates to the existing DataPreparationLayer."""

    def __init__(
        self, max_retries: int = 0, retry_base_delay: float = 1.0
    ) -> None:
        self._layer = DataPreparationLayer(
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )

    def prepare_data(
        self, symbol: str, requirements: object, *, allow_partial: bool = True
    ) -> object:
        return self._layer.prepare_data(
            symbol, requirements, allow_partial=allow_partial
        )

    def prepare_data_bulk(
        self,
        symbols: list[str],
        requirements: object,
        *,
        allow_partial: bool = True,
        batch_only_prices: bool = False,
        batch_only_fundamentals: bool = False,
    ) -> dict[str, object]:
        return self._layer.prepare_data_bulk(
            symbols,
            requirements,
            allow_partial=allow_partial,
            batch_only_prices=batch_only_prices,
            batch_only_fundamentals=batch_only_fundamentals,
        )
