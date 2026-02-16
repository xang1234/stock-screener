from datetime import date, timedelta
from uuid import uuid4

from app.models.industry import IBDGroupRank
from app.services.ibd_group_rank_service import IBDGroupRankService


def _add_rank(session, group, rank_date, rank):
    session.add(
        IBDGroupRank(
            industry_group=group,
            date=rank_date,
            rank=rank,
            avg_rs_rating=50.0,
            num_stocks=10,
            num_stocks_rs_above_80=2,
            top_symbol="TEST",
            top_rs_rating=90.0,
        )
    )
    session.flush()


def test_get_historical_rank_picks_closest_date(db_session):
    service = IBDGroupRankService.get_instance()
    group = f"TEST_GROUP_UNIT_{uuid4().hex}"
    current_date = date(2024, 1, 22)

    try:
        # target_date = current_date - 7 days = Jan 15
        # Record 2 days before target (Jan 13, rank 10) — closer
        # Record 3 days after target (Jan 18, rank 20) — further
        _add_rank(db_session, group, date(2024, 1, 13), 10)
        _add_rank(db_session, group, date(2024, 1, 18), 20)

        result = service._get_historical_ranks_batch(
            db_session, [group], current_date, {'1w': 7}
        )
        assert result[(group, '1w')] == 10
    finally:
        db_session.rollback()


def test_get_historical_rank_prefers_earlier_on_tie(db_session):
    service = IBDGroupRankService.get_instance()
    group = f"TEST_GROUP_UNIT_{uuid4().hex}"
    current_date = date(2024, 1, 22)

    try:
        # target_date = current_date - 7 days = Jan 15
        # Record 2 days before target (Jan 13, rank 11) — equidistant, earlier
        # Record 2 days after target (Jan 17, rank 99) — equidistant, later
        _add_rank(db_session, group, date(2024, 1, 13), 11)
        _add_rank(db_session, group, date(2024, 1, 17), 99)

        result = service._get_historical_ranks_batch(
            db_session, [group], current_date, {'1w': 7}
        )
        assert result[(group, '1w')] == 11
    finally:
        db_session.rollback()
