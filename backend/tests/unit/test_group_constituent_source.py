from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.scan_result import Scan, ScanResult
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse
from app.services.group_constituent_source import GroupConstituentSource


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
            Scan.__table__,
            ScanResult.__table__,
            StockUniverse.__table__,
            StockFundamental.__table__,
        ],
    )
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _add_scan(
    session,
    *,
    scan_id: str,
    market: str = "US",
    completed_at: datetime,
    feature_run_id: int | None = None,
):
    session.add(
        Scan(
            scan_id=scan_id,
            status="completed",
            universe_market=market,
            completed_at=completed_at,
            feature_run_id=feature_run_id,
        )
    )


def _add_feature_run(session, *, run_id: int, as_of_date: date, market: str = "US"):
    session.add(
        FeatureRun(
            id=run_id,
            as_of_date=as_of_date,
            run_type="daily_snapshot",
            status="published",
            published_at=datetime.combine(as_of_date, datetime.min.time()),
            config_json={"universe": {"market": market}},
        )
    )


def _add_feature_row(
    session,
    *,
    run_id: int,
    symbol: str,
    as_of_date: date,
    industry_group: str,
):
    session.add(
        StockFeatureDaily(
            run_id=run_id,
            symbol=symbol,
            as_of_date=as_of_date,
            composite_score=88.0,
            details_json={
                "current_price": 123.45,
                "rs_rating": 96.0,
                "ibd_industry_group": industry_group,
                "price_sparkline_data": [1.0, 1.2],
                "rs_sparkline_data": [1.0, 1.3],
            },
        )
    )


def _add_legacy_scan_result(
    session,
    *,
    scan_id: str,
    symbol: str,
    industry_group: str,
):
    session.add(
        ScanResult(
            scan_id=scan_id,
            symbol=symbol,
            price=42.0,
            rs_rating=96.0,
            ibd_industry_group=industry_group,
            price_sparkline_data=[1.0, 1.1],
            rs_sparkline_data=[1.0, 1.2],
        )
    )


def test_feature_run_with_empty_group_does_not_fallback_to_legacy():
    db = _make_session()
    group = "Software"
    as_of_date = date(2026, 6, 24)
    try:
        _add_feature_run(db, run_id=88, as_of_date=as_of_date)
        _add_feature_row(
            db,
            run_id=88,
            symbol="OTHER",
            as_of_date=as_of_date,
            industry_group="Other",
        )
        _add_scan(
            db,
            scan_id="legacy-us-scan",
            market="US",
            completed_at=datetime(2026, 6, 23, 22, 0, 0),
        )
        _add_legacy_scan_result(
            db,
            scan_id="legacy-us-scan",
            symbol="LEGACY",
            industry_group=group,
        )
        db.commit()

        items = GroupConstituentSource().get_constituent_items(
            db,
            group,
            market="US",
            as_of_date=as_of_date,
        )

        assert items == ()
    finally:
        db.rollback()
        db.close()


def test_feature_run_market_metadata_controls_feature_constituent_source():
    db = _make_session()
    group = "Software"
    as_of_date = date(2026, 6, 24)
    try:
        _add_feature_run(db, run_id=88, as_of_date=as_of_date, market="HK")
        _add_scan(
            db,
            scan_id="mismatched-feature-scan",
            market="US",
            completed_at=datetime(2026, 6, 24, 22, 0, 0),
            feature_run_id=88,
        )
        _add_feature_row(
            db,
            run_id=88,
            symbol="HKROW",
            as_of_date=as_of_date,
            industry_group=group,
        )
        _add_scan(
            db,
            scan_id="legacy-us-scan",
            market="US",
            completed_at=datetime(2026, 6, 23, 22, 0, 0),
        )
        _add_legacy_scan_result(
            db,
            scan_id="legacy-us-scan",
            symbol="LEGACY",
            industry_group=group,
        )
        db.commit()

        items = GroupConstituentSource().get_constituent_items(
            db,
            group,
            market="US",
            as_of_date=as_of_date,
        )

        assert [item.symbol for item in items] == ["LEGACY"]
    finally:
        db.rollback()
        db.close()


def test_legacy_scan_constituents_are_market_scoped_and_include_sparklines():
    db = _make_session()
    group = "Software"
    try:
        _add_scan(
            db,
            scan_id="us-scan",
            market="US",
            completed_at=datetime(2026, 6, 24, 22, 0, 0),
        )
        _add_scan(
            db,
            scan_id="hk-scan",
            market="HK",
            completed_at=datetime(2026, 6, 26, 22, 0, 0),
        )
        db.add_all(
            [
                StockUniverse(symbol="USWIN", name="US Winner", market="US"),
                StockUniverse(symbol="HKNEW", name="HK Newer", market="HK"),
            ]
        )
        _add_legacy_scan_result(
            db,
            scan_id="us-scan",
            symbol="USWIN",
            industry_group=group,
        )
        _add_legacy_scan_result(
            db,
            scan_id="hk-scan",
            symbol="HKNEW",
            industry_group=group,
        )
        db.commit()

        items = GroupConstituentSource().get_constituent_items(
            db,
            group,
            market="US",
            as_of_date=date(2026, 6, 24),
        )

        assert [item.symbol for item in items] == ["USWIN"]
        assert items[0].extended_fields["company_name"] == "US Winner"
        assert items[0].extended_fields["price_sparkline_data"] == [1.0, 1.1]
        assert items[0].extended_fields["rs_sparkline_data"] == [1.0, 1.2]
    finally:
        db.rollback()
        db.close()


def test_legacy_scan_selection_honors_as_of_date():
    db = _make_session()
    group = "Software"
    try:
        _add_scan(
            db,
            scan_id="past-us-scan",
            market="US",
            completed_at=datetime(2026, 6, 24, 22, 0, 0),
        )
        _add_scan(
            db,
            scan_id="future-us-scan",
            market="US",
            completed_at=datetime(2026, 6, 25, 1, 0, 0),
        )
        db.add_all(
            [
                StockUniverse(symbol="PAST", name="Past Corp", market="US"),
                StockUniverse(symbol="FUTURE", name="Future Corp", market="US"),
            ]
        )
        _add_legacy_scan_result(
            db,
            scan_id="past-us-scan",
            symbol="PAST",
            industry_group=group,
        )
        _add_legacy_scan_result(
            db,
            scan_id="future-us-scan",
            symbol="FUTURE",
            industry_group=group,
        )
        db.commit()

        items = GroupConstituentSource().get_constituent_items(
            db,
            group,
            market="US",
            as_of_date=date(2026, 6, 24),
        )

        assert [item.symbol for item in items] == ["PAST"]
    finally:
        db.rollback()
        db.close()
