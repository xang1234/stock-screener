"""Resolve scanner RS values under the active Market RS formula."""

from __future__ import annotations

from collections.abc import Callable

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.scanners.base_screener import StockData


class StockRsUnavailable(RuntimeError):
    """The active RS mode cannot produce a rating for this stock."""


class CanonicalStockRsUnavailable(StockRsUnavailable):
    """The active canonical formula has no eligible rating for this stock."""


class LegacyStockRsUnavailable(StockRsUnavailable):
    """The legacy calculation lacks required local inputs."""


def resolve_stock_rs(
    stock_data: StockData,
    legacy_factory: Callable[[], dict[str, float | int]],
) -> dict[str, float | int]:
    """Return canonical ratings or explicitly use the legacy calculation path.

    Balanced mode fails closed when the symbol is absent from its run's single
    eligible universe. This prevents an ineligible stock from being calculated
    against a different, scan-local comparison set.
    """
    if stock_data.canonical_rs_ratings is not None:
        return dict(stock_data.canonical_rs_ratings)
    if stock_data.rs_formula_version == BALANCED_RS_FORMULA_VERSION:
        raise CanonicalStockRsUnavailable(
            f"Canonical RS is unavailable for {stock_data.symbol} in the active "
            f"{BALANCED_RS_FORMULA_VERSION} eligible universe"
        )
    return dict(legacy_factory())
