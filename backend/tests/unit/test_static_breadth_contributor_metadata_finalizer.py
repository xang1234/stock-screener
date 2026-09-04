from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.breadth_contributor import (
    MarketBreadthContributor,
    MarketBreadthContributorSnapshot,
)
from app.models.market_breadth import MarketBreadth
from app.services.breadth.types import BreadthContributorMetadata
from app.services.static_breadth_contributor_metadata_contract import (
    STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION,
    FrozenBreadthContributorMetadata,
    FrozenBreadthContributorSession,
    StaticBreadthContributorMetadataState,
    read_static_breadth_contributor_metadata,
    write_static_breadth_contributor_metadata,
)
from app.services.static_breadth_contributor_metadata_finalizer import (
    StaticBreadthContributorMetadataCoverageError,
    StaticBreadthContributorMetadataFinalizer,
)


def _db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            MarketBreadth.__table__,
            MarketBreadthContributorSnapshot.__table__,
            MarketBreadthContributor.__table__,
        ],
    )
    return sessionmaker(bind=engine)()


def _seed(db, calculation_date: date, symbol: str = "AAA") -> None:
    db.add(
        MarketBreadth(
            market="US",
            date=calculation_date,
            calculation_revision=3,
            stocks_up_4pct=1,
            stocks_down_4pct=0,
            stocks_up_25pct_quarter=0,
            stocks_down_25pct_quarter=0,
            stocks_up_25pct_month=0,
            stocks_down_25pct_month=0,
            stocks_up_50pct_month=0,
            stocks_down_50pct_month=0,
            stocks_up_13pct_34days=0,
            stocks_down_13pct_34days=0,
            atr_10x_extension_count=0,
            total_stocks_scanned=1,
            contributor_calculation_signature=f"signature-{calculation_date}",
        )
    )
    db.add(
        MarketBreadthContributorSnapshot(
            market="US",
            date=calculation_date,
            calculation_revision=3,
            schema_id="breadth-contributors-v1",
            contributors=[
                MarketBreadthContributor(
                    symbol=symbol,
                    company_name=None,
                    ibd_industry_group="No Group",
                    daily_change_pct=5.25,
                    signals_json={"up_4pct": 5.25},
                )
            ],
        )
    )
    db.commit()


def _loader(company: str | None, group: str):
    def current(_db, _market, symbols):
        return MappingProxyType(
            {
                symbol: BreadthContributorMetadata(
                    company_name=company,
                    ibd_industry_group=group,
                )
                for symbol in symbols
            }
        )

    return SimpleNamespace(current=current)


def _restored_state(path: Path, calculation_date: date) -> None:
    write_static_breadth_contributor_metadata(
        path,
        StaticBreadthContributorMetadataState(
            schema_version=STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION,
            market="US",
            generated_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
            sessions=(
                FrozenBreadthContributorSession(
                    date=calculation_date,
                    contributors=(
                        FrozenBreadthContributorMetadata(
                            symbol="AAA",
                            company_name="Frozen Alpha",
                            ibd_industry_group="Frozen Group",
                        ),
                    ),
                ),
            ),
        ),
    )


def test_finalizer_preserves_restored_metadata_and_bootstraps_new_sessions(tmp_path):
    db = _db_session()
    old_date = date(2026, 8, 28)
    new_date = date(2026, 8, 31)
    _seed(db, old_date)
    _seed(db, new_date)
    source = tmp_path / "restored.json.gz"
    output = tmp_path / "current.json.gz"
    _restored_state(source, old_date)

    report = StaticBreadthContributorMetadataFinalizer(
        db,
        metadata_loader=_loader("Current Alpha", "Current Group"),
    ).finalize(
        market="US",
        source_path=source,
        output_path=output,
        source_status="restored",
    )

    rows = {
        row.snapshot.date: row
        for row in db.query(MarketBreadthContributor).all()
    }
    assert rows[old_date].company_name == "Frozen Alpha"
    assert rows[old_date].ibd_industry_group == "Frozen Group"
    assert rows[new_date].company_name == "Current Alpha"
    assert rows[new_date].ibd_industry_group == "Current Group"
    assert report.restored_contributors == 1
    assert report.bootstrapped_contributors == 1
    state = read_static_breadth_contributor_metadata(output, expected_market="US")
    assert [session.date for session in state.sessions] == [new_date, old_date]


def test_finalizer_retains_latest_twenty_and_only_changes_display_metadata(tmp_path):
    db = _db_session()
    start = date(2026, 8, 1)
    for offset in range(21):
        _seed(db, start + timedelta(days=offset))
    before = [
        (row.date, row.stocks_up_4pct, row.contributor_calculation_signature)
        for row in db.query(MarketBreadth).order_by(MarketBreadth.date).all()
    ]

    StaticBreadthContributorMetadataFinalizer(
        db,
        metadata_loader=_loader("Alpha", "Semiconductors"),
    ).finalize(
        market="US",
        source_path=tmp_path / "missing.json.gz",
        output_path=tmp_path / "current.json.gz",
        source_status="missing",
    )

    after = [
        (row.date, row.stocks_up_4pct, row.contributor_calculation_signature)
        for row in db.query(MarketBreadth).order_by(MarketBreadth.date).all()
    ]
    state = read_static_breadth_contributor_metadata(
        tmp_path / "current.json.gz", expected_market="US"
    )
    assert before == after
    assert len(state.sessions) == 20
    assert state.sessions[0].date == start + timedelta(days=20)
    assert state.sessions[-1].date == start + timedelta(days=1)
    contributor = db.query(MarketBreadthContributor).first()
    assert contributor.daily_change_pct == 5.25
    assert contributor.signals_json == {"up_4pct": 5.25}


@pytest.mark.parametrize(
    ("company", "group", "expected_message"),
    [
        (None, "Semiconductors", "company name"),
        ("Alpha", "No Group", "industry group"),
    ],
)
def test_finalizer_rejects_unusable_nonempty_metadata_and_rolls_back(
    tmp_path,
    company,
    group,
    expected_message,
):
    db = _db_session()
    _seed(db, date(2026, 8, 31))

    with pytest.raises(
        StaticBreadthContributorMetadataCoverageError, match=expected_message
    ):
        StaticBreadthContributorMetadataFinalizer(
            db,
            metadata_loader=_loader(company, group),
        ).finalize(
            market="US",
            source_path=tmp_path / "missing.json.gz",
            output_path=tmp_path / "current.json.gz",
            source_status="missing",
        )

    contributor = db.query(MarketBreadthContributor).one()
    assert contributor.company_name is None
    assert contributor.ibd_industry_group == "No Group"
    assert not (tmp_path / "current.json.gz").exists()


def test_finalizer_writes_valid_empty_state(tmp_path):
    db = _db_session()
    output = tmp_path / "current.json.gz"

    report = StaticBreadthContributorMetadataFinalizer(
        db,
        metadata_loader=_loader(None, "No Group"),
    ).finalize(
        market="US",
        source_path=tmp_path / "missing.json.gz",
        output_path=output,
        source_status="missing",
    )

    assert report.contributors == 0
    assert read_static_breadth_contributor_metadata(
        output, expected_market="US"
    ).sessions == ()
