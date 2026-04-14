"""End-to-end integration: seed a CSV, resolve a universe, get the right symbols.

Exercises the full Asia-index vertical slice without the scan pipeline:

1. seed_from_csv populates stock_universe_index_membership from a CSV.
2. resolve_symbols(UniverseDefinition(type=INDEX, index=HSI)) routes
   through the service layer and returns the seeded constituents
   intersected with the active universe.

Complements the unit tests in:
- ``test_index_membership_seeder.py`` (seeder CSV → DB shape)
- ``test_universe_resolver_asia_indices.py`` (resolver dispatch with
  mocked service)
- ``test_stock_universe_service_index_membership.py`` (service-layer
  filter with hand-seeded DB)

This file is the "all the above wired together" sanity test.
"""

from __future__ import annotations

from pathlib import Path

from app.models.stock_universe import StockUniverse
from app.schemas.universe import IndexName, UniverseDefinition, UniverseType
from app.services.index_membership_seeder import seed_from_csv
from app.services.universe_resolver import resolve_symbols


def _seed_universe(universe_session, symbols: list[tuple[str, str, bool]]) -> None:
    """Add StockUniverse rows. Each tuple is (symbol, market, is_active)."""
    for symbol, market, is_active in symbols:
        universe_session.add(
            StockUniverse(
                symbol=symbol,
                name=f"Stub {symbol}",
                market=market,
                is_active=is_active,
            )
        )
    universe_session.commit()


def _write_hsi_csv(tmp_path: Path) -> Path:
    path = tmp_path / "hsi.csv"
    path.write_text(
        "symbol,name\n"
        "0700.HK,Tencent\n"
        "0005.HK,HSBC\n"
        "0388.HK,HKEX\n"
        "DEAD.HK,Delisted Corp\n",
        encoding="utf-8",
    )
    return path


class TestResolverWithSeededMembership:
    def test_hsi_universe_returns_only_active_seeded_symbols(
        self, universe_session, tmp_path
    ):
        # 4 symbols in universe: 3 active HK stocks + 1 deactivated.
        _seed_universe(
            universe_session,
            [
                ("0700.HK", "HK", True),
                ("0005.HK", "HK", True),
                ("0388.HK", "HK", True),
                ("DEAD.HK", "HK", False),  # in CSV but inactive
                ("9999.HK", "HK", True),   # active but not in HSI CSV
            ],
        )
        seed_from_csv(
            universe_session,
            _write_hsi_csv(tmp_path),
            index_name="HSI",
            as_of_date="2025-05-01",
        )

        universe_def = UniverseDefinition(
            type=UniverseType.INDEX, index=IndexName.HSI
        )
        symbols = resolve_symbols(universe_session, universe_def)

        # DEAD.HK filtered out by active_filter; 9999.HK filtered out by
        # membership join; 0700/0005/0388 come through.
        assert set(symbols) == {"0700.HK", "0005.HK", "0388.HK"}

    def test_unseeded_nikkei_universe_returns_empty(self, universe_session, tmp_path):
        # Seed HSI but NOT Nikkei. Requesting Nikkei should fail-closed.
        _seed_universe(universe_session, [("6758.T", "JP", True)])
        seed_from_csv(
            universe_session,
            _write_hsi_csv(tmp_path),
            index_name="HSI",
            as_of_date="2025-05-01",
        )

        universe_def = UniverseDefinition(
            type=UniverseType.INDEX, index=IndexName.NIKKEI225
        )
        assert resolve_symbols(universe_session, universe_def) == []

    def test_limit_propagates_through_resolver(self, universe_session, tmp_path):
        _seed_universe(
            universe_session,
            [("0700.HK", "HK", True), ("0005.HK", "HK", True), ("0388.HK", "HK", True)],
        )
        seed_from_csv(
            universe_session,
            _write_hsi_csv(tmp_path),
            index_name="HSI",
            as_of_date="2025-05-01",
        )

        universe_def = UniverseDefinition(
            type=UniverseType.INDEX, index=IndexName.HSI
        )
        symbols = resolve_symbols(universe_session, universe_def, limit=2)
        assert len(symbols) == 2
