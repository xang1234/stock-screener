from datetime import date
from types import SimpleNamespace

import pytest

from app.services.group_ranking_payloads import (
    annotate_top_symbol_names,
    group_snapshot_metadata,
    rank_record_payload,
)
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.models.industry import IBDGroupRank
from app.models.stock_universe import StockUniverse


class _MetadataDb:
    def __init__(self, run=None) -> None:
        self.run = run

    def get(self, model, run_id):  # noqa: ANN001, ARG002
        return self.run


def test_annotate_top_symbol_names_batches_known_and_unknown_symbols(db_session):
    db_session.add(StockUniverse(symbol="AAA", name="A Corp", market="US"))
    db_session.commit()
    rows = [
        {"top_symbol": "AAA", "top_symbol_name": None},
        {"top_symbol": "MISSING", "top_symbol_name": "stale"},
    ]

    annotate_top_symbol_names(db_session, rows)

    assert rows == [
        {"top_symbol": "AAA", "top_symbol_name": "A Corp"},
        {"top_symbol": "MISSING", "top_symbol_name": None},
    ]


def test_group_snapshot_metadata_rejects_rows_missing_audit_fields():
    with pytest.raises(RuntimeError, match="mixes canonical RS sources"):
        group_snapshot_metadata(
            _MetadataDb(),
            market="US",
            rankings=[
                {
                    "industry_group": "Software",
                    "date": "2026-04-10",
                    "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
                    "market_rs_run_id": 42,
                },
                {
                    "industry_group": "Semiconductors",
                    "date": "2026-04-10",
                    "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
                    "market_rs_run_id": None,
                },
            ],
        )


def test_group_snapshot_metadata_requires_the_referenced_completed_run():
    rows = [
        {
            "industry_group": "Software",
            "date": "2026-04-10",
            "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            "market_rs_run_id": 42,
        }
    ]
    with pytest.raises(RuntimeError, match="does not match"):
        group_snapshot_metadata(_MetadataDb(), market="US", rankings=rows)

    metadata = group_snapshot_metadata(
        _MetadataDb(
            SimpleNamespace(
                market="US",
                formula_version=BALANCED_RS_FORMULA_VERSION,
                as_of_date=date(2026, 4, 10),
                status="completed",
                eligible_symbol_count=5000,
            )
        ),
        market="us",
        rankings=rows,
    )
    assert metadata == {
        "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
        "rs_as_of_date": "2026-04-10",
        "rs_universe_size": 5000,
    }


def test_rank_record_payload_includes_canonical_components_and_audit_fields():
    ranking = IBDGroupRank(
        market="US",
        industry_group="Software",
        date=date(2026, 4, 10),
        rank=1,
        avg_rs_rating=87.25,
        avg_rs_rating_1m=42.5,
        avg_rs_rating_3m=91.5,
        num_stocks=4,
        num_stocks_rs_above_80=3,
        top_symbol="AAA",
        top_rs_rating=99,
        rs_formula_version=BALANCED_RS_FORMULA_VERSION,
        market_rs_run_id=42,
    )

    payload = rank_record_payload(ranking, pct_rs_above_80=75.0)

    assert payload["avg_rs_rating_1m"] == 42.5
    assert payload["avg_rs_rating_3m"] == 91.5
    assert payload["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
    assert payload["market_rs_run_id"] == 42
