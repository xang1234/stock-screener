from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.industry import IBDIndustryGroup
from app.services.ibd_industry_service import IBDIndustryService


def _make_session_factory(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'ibd-industry.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def test_seed_if_empty_loads_mappings_when_table_empty(tmp_path):
    session_factory = _make_session_factory(tmp_path)
    csv_path = tmp_path / "ibd_industry_group.csv"
    csv_path.write_text(
        "AAPL,Computer-Hardware/Peripherals\nMSFT,Computer-Software/Desktop\n",
        encoding="utf-8",
    )

    with session_factory() as db:
        loaded = IBDIndustryService.seed_if_empty(db, str(csv_path))
        rows = db.query(IBDIndustryGroup).order_by(IBDIndustryGroup.symbol).all()

    assert loaded == 2
    assert [(row.symbol, row.industry_group) for row in rows] == [
        ("AAPL", "Computer-Hardware/Peripherals"),
        ("MSFT", "Computer-Software/Desktop"),
    ]


def test_seed_if_empty_is_noop_when_rows_already_exist(tmp_path):
    session_factory = _make_session_factory(tmp_path)
    csv_path = tmp_path / "ibd_industry_group.csv"
    csv_path.write_text("MSFT,Should-Not-Replace\n", encoding="utf-8")

    with session_factory() as db:
        db.add(IBDIndustryGroup(symbol="AAPL", industry_group="Existing-Group"))
        db.commit()

        loaded = IBDIndustryService.seed_if_empty(db, str(csv_path))
        rows = db.query(IBDIndustryGroup).order_by(IBDIndustryGroup.symbol).all()

    assert loaded == 0
    assert [(row.symbol, row.industry_group) for row in rows] == [
        ("AAPL", "Existing-Group"),
    ]
