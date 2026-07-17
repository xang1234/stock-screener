from datetime import date
from unittest.mock import Mock

from sqlalchemy import event

from app.models.industry import IBDGroupRank
from app.services.group_ranking_repository import (
    GroupRankingRepository,
)


def _ranking(group: str, *, rank: int) -> dict[str, object]:
    return {
        "industry_group": group,
        "rank": rank,
        "avg_rs_rating": 80.0,
        "median_rs_rating": 80.0,
        "weighted_avg_rs_rating": 80.0,
        "rs_std_dev": 0.0,
        "num_stocks": 3,
        "num_stocks_rs_above_80": 1,
        "top_symbol": "AAA",
        "top_rs_rating": 90.0,
    }


def _seed_rank(
    db_session,
    *,
    market: str,
    calculation_date: date,
    group: str = "Software",
    rank: int = 1,
) -> None:
    db_session.add(
        IBDGroupRank(
            market=market,
            date=calculation_date,
            industry_group=group,
            rank=rank,
            avg_rs_rating=80.0,
            median_rs_rating=80.0,
            weighted_avg_rs_rating=80.0,
            rs_std_dev=0.0,
            num_stocks=3,
            num_stocks_rs_above_80=1,
            top_symbol="AAA",
            top_rs_rating=90.0,
        )
    )


def test_store_rankings_does_not_commit(db_session, monkeypatch):
    repository = GroupRankingRepository()
    commit = Mock(
        side_effect=AssertionError("repository must not commit")
    )
    monkeypatch.setattr(db_session, "commit", commit)

    repository.store_rankings(
        db_session,
        calculation_date=date(2026, 3, 20),
        rankings=(_ranking("Software", rank=1),),
        market="US",
    )

    commit.assert_not_called()


def test_store_rankings_bulk_loads_existing_rows_once_for_sqlite_fallback(
    db_session,
):
    repository = GroupRankingRepository()
    engine = db_session.get_bind()
    query_counts = {"select": 0}

    def count_selects(
        _conn,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        if statement.lstrip().upper().startswith("SELECT"):
            query_counts["select"] += 1

    _seed_rank(
        db_session,
        market="US",
        calculation_date=date(2026, 3, 20),
        rank=9,
    )
    db_session.commit()

    try:
        event.listen(
            engine,
            "before_cursor_execute",
            count_selects,
        )
        repository.store_rankings(
            db_session,
            calculation_date=date(2026, 3, 20),
            rankings=(
                _ranking("Software", rank=1),
                _ranking("Semiconductors", rank=2),
            ),
            market="US",
        )
        db_session.flush()
        event.remove(
            engine,
            "before_cursor_execute",
            count_selects,
        )

        rows = (
            db_session.query(IBDGroupRank)
            .order_by(IBDGroupRank.rank)
            .all()
        )

        assert query_counts["select"] == 1
        assert len(rows) == 2
        assert rows[0].industry_group == "Software"
        assert rows[1].industry_group == "Semiconductors"
    finally:
        try:
            event.remove(
                engine,
                "before_cursor_execute",
                count_selects,
            )
        except Exception:
            pass


def test_delete_range_is_market_scoped(db_session):
    _seed_rank(
        db_session,
        market="US",
        calculation_date=date(2026, 3, 20),
    )
    _seed_rank(
        db_session,
        market="JP",
        calculation_date=date(2026, 3, 20),
    )
    db_session.commit()

    deleted = GroupRankingRepository().delete_range(
        db_session,
        start_date=date(2026, 3, 20),
        end_date=date(2026, 3, 20),
        market="US",
    )

    assert deleted == 1
    assert (
        db_session.query(IBDGroupRank)
        .filter_by(market="JP")
        .count()
        == 1
    )


def test_current_rank_rows_select_latest_or_explicit_market_date(
    db_session,
):
    _seed_rank(
        db_session,
        market="US",
        calculation_date=date(2026, 3, 19),
        rank=5,
    )
    _seed_rank(
        db_session,
        market="US",
        calculation_date=date(2026, 3, 20),
        rank=1,
    )
    _seed_rank(
        db_session,
        market="JP",
        calculation_date=date(2026, 3, 21),
        rank=1,
    )
    db_session.commit()
    repository = GroupRankingRepository()

    latest = repository.current_rank_rows(
        db_session,
        limit=10,
        market="US",
        calculation_date=None,
    )
    explicit = repository.current_rank_rows(
        db_session,
        limit=10,
        market="US",
        calculation_date=date(2026, 3, 19),
    )

    assert [row.date for row in latest] == [date(2026, 3, 20)]
    assert [row.rank for row in explicit] == [5]


def test_historical_ranks_batch_picks_closest_and_prefers_earlier_tie(
    db_session,
):
    current_date = date(2024, 1, 22)
    _seed_rank(
        db_session,
        market="US",
        calculation_date=date(2024, 1, 13),
        group="Software",
        rank=11,
    )
    _seed_rank(
        db_session,
        market="US",
        calculation_date=date(2024, 1, 17),
        group="Software",
        rank=99,
    )
    db_session.commit()

    result = GroupRankingRepository().historical_ranks_batch(
        db_session,
        group_names=("Software",),
        current_date=current_date,
        period_days={"1w": 7},
        market="US",
    )

    assert result[("Software", "1w")] == 11
