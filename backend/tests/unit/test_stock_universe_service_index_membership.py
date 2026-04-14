"""``get_active_symbols`` routes per-index to the right filter.

SP500 uses the legacy ``is_sp500`` column; HSI/NIKKEI225/TAIEX resolve via
the ``stock_universe_index_membership`` table. Unknown / unseeded index
names return an empty list (fail-closed so callers don't accidentally scan
a whole market when membership data hasn't landed yet).
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
import app.models.stock_universe  # noqa: F401
import app.models.stock  # noqa: F401
from app.models.stock_universe import StockUniverse, StockUniverseIndexMembership
from app.wiring.bootstrap import get_stock_universe_service


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


def _add_symbol(session, symbol: str, market: str, *, is_sp500: bool = False) -> None:
    session.add(
        StockUniverse(
            symbol=symbol,
            name=f"Stub {symbol}",
            market=market,
            is_active=True,
            is_sp500=is_sp500,
        )
    )


def _add_membership(session, symbol: str, index_name: str) -> None:
    session.add(
        StockUniverseIndexMembership(
            symbol=symbol,
            index_name=index_name,
            as_of_date="2026-04-14",
            source="test",
        )
    )


class TestIndexMembershipRouting:
    def test_sp500_resolves_via_is_sp500_column(self, session):
        _add_symbol(session, "AAPL", market="US", is_sp500=True)
        _add_symbol(session, "TSLA", market="US", is_sp500=False)
        session.commit()

        result = get_stock_universe_service().get_active_symbols(
            session, index_name="SP500"
        )
        assert result == ["AAPL"]

    def test_sp500_only_legacy_flag_still_works(self, session):
        # Backward compat: existing callers of sp500_only=True keep working.
        _add_symbol(session, "AAPL", market="US", is_sp500=True)
        session.commit()

        result = get_stock_universe_service().get_active_symbols(
            session, sp500_only=True
        )
        assert result == ["AAPL"]

    def test_hsi_resolves_via_membership_table(self, session):
        _add_symbol(session, "0700.HK", market="HK")
        _add_symbol(session, "0005.HK", market="HK")
        _add_symbol(session, "9999.HK", market="HK")  # not in HSI
        _add_membership(session, "0700.HK", "HSI")
        _add_membership(session, "0005.HK", "HSI")
        session.commit()

        result = get_stock_universe_service().get_active_symbols(
            session, index_name="HSI"
        )
        assert set(result) == {"0700.HK", "0005.HK"}

    def test_nikkei225_resolves_via_membership_table(self, session):
        _add_symbol(session, "6758.T", market="JP")
        _add_symbol(session, "7203.T", market="JP")
        _add_membership(session, "6758.T", "NIKKEI225")
        session.commit()

        result = get_stock_universe_service().get_active_symbols(
            session, index_name="NIKKEI225"
        )
        assert result == ["6758.T"]

    def test_unseeded_index_returns_empty_fail_closed(self, session):
        # Data source for TAIEX hasn't landed yet → no rows in membership
        # table → we return no symbols rather than leaking a whole-market scan.
        _add_symbol(session, "2330.TW", market="TW")
        session.commit()

        result = get_stock_universe_service().get_active_symbols(
            session, index_name="TAIEX"
        )
        assert result == []

    def test_inactive_symbols_excluded_even_if_in_membership(self, session):
        # Membership table doesn't bypass the active_filter — a delisted
        # constituent still drops out of scan sets.
        session.add(
            StockUniverse(
                symbol="ZZZZ.HK",
                name="Delisted",
                market="HK",
                is_active=False,
            )
        )
        _add_membership(session, "ZZZZ.HK", "HSI")
        session.commit()

        result = get_stock_universe_service().get_active_symbols(
            session, index_name="HSI"
        )
        assert result == []

    def test_unknown_index_name_returns_empty(self, session):
        _add_symbol(session, "AAPL", market="US", is_sp500=True)
        session.commit()

        # A typo or a future index that isn't in the membership table yet
        # should not silently fall through to "all active symbols".
        result = get_stock_universe_service().get_active_symbols(
            session, index_name="FTSE100"
        )
        assert result == []

    def test_index_name_is_case_insensitive(self, session):
        _add_symbol(session, "0700.HK", market="HK")
        _add_membership(session, "0700.HK", "HSI")
        session.commit()

        lower = get_stock_universe_service().get_active_symbols(
            session, index_name="hsi"
        )
        assert lower == ["0700.HK"]
