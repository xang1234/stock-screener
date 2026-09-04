from __future__ import annotations

import gzip
import json
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.services.static_breadth_contributor_metadata_contract import (
    STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION,
    FrozenBreadthContributorMetadata,
    FrozenBreadthContributorSession,
    StaticBreadthContributorMetadataBundleError,
    StaticBreadthContributorMetadataState,
    build_static_breadth_contributor_metadata_plan,
    read_static_breadth_contributor_metadata,
    write_static_breadth_contributor_metadata,
)


def _contributor(
    symbol: str = "AAA",
    *,
    company_name: str | None = "Alpha Inc",
    group: str = "Software",
) -> FrozenBreadthContributorMetadata:
    return FrozenBreadthContributorMetadata(
        symbol=symbol,
        company_name=company_name,
        ibd_industry_group=group,
    )


def _state(*sessions: FrozenBreadthContributorSession, market: str = "US"):
    return StaticBreadthContributorMetadataState(
        schema_version=STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION,
        market=market,
        generated_at=datetime(2026, 8, 31, 4, tzinfo=UTC),
        sessions=sessions,
    )


def test_metadata_state_round_trips_as_deterministic_gzip(tmp_path):
    state = _state(
        FrozenBreadthContributorSession(
            date=date(2026, 8, 28),
            contributors=(
                _contributor(
                    "BTAI",
                    company_name="BioXcel Therapeutics Inc",
                    group="Medical-Biomed/Biotech",
                ),
            ),
        )
    )
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"

    write_static_breadth_contributor_metadata(first, state)
    write_static_breadth_contributor_metadata(second, state)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[:2] == b"\x1f\x8b"
    assert read_static_breadth_contributor_metadata(
        first,
        expected_market="us",
    ) == state


def test_metadata_plan_uses_market_scoped_deterministic_paths(tmp_path):
    plan = build_static_breadth_contributor_metadata_plan(
        market=" us ",
        directory=tmp_path,
    )

    assert plan.enabled is True
    assert plan.market == "US"
    assert plan.asset_name == "breadth-contributor-metadata-us.json.gz"
    assert plan.previous_asset_name == (
        "breadth-contributor-metadata-us.previous.json.gz"
    )
    assert plan.source_path == tmp_path / plan.asset_name
    assert plan.previous_path == tmp_path / plan.previous_asset_name
    assert plan.output_path == tmp_path / "current" / plan.asset_name
    assert plan.as_dict() == {
        "enabled": True,
        "market": "US",
        "asset_name": plan.asset_name,
        "previous_asset_name": plan.previous_asset_name,
        "source_path": str(plan.source_path),
        "previous_path": str(plan.previous_path),
        "output_path": str(plan.output_path),
    }


def test_metadata_normalizes_symbols_text_and_blank_groups():
    contributor = _contributor(
        " aaa ",
        company_name=" Alpha Inc ",
        group="   ",
    )

    assert contributor.symbol == "AAA"
    assert contributor.company_name == "Alpha Inc"
    assert contributor.ibd_industry_group == "No Group"


@pytest.mark.parametrize("invalid_name", [123, [], {"name": "Alpha"}])
def test_metadata_rejects_non_string_company_names(invalid_name):
    with pytest.raises(ValidationError, match="company_name"):
        FrozenBreadthContributorMetadata(
            symbol="AAA",
            company_name=invalid_name,
            ibd_industry_group="Software",
        )


@pytest.mark.parametrize("invalid_group", [None, 123, [], {"group": "Software"}])
def test_metadata_rejects_non_string_industry_groups(invalid_group):
    with pytest.raises(ValidationError, match="ibd_industry_group"):
        FrozenBreadthContributorMetadata(
            symbol="AAA",
            company_name=None,
            ibd_industry_group=invalid_group,
        )


def test_metadata_sessions_must_be_newest_first_and_unique():
    newer = FrozenBreadthContributorSession(
        date=date(2026, 8, 28),
        contributors=(_contributor(),),
    )
    older = FrozenBreadthContributorSession(
        date=date(2026, 8, 27),
        contributors=(_contributor(),),
    )

    with pytest.raises(ValidationError, match="newest-first"):
        _state(older, newer)
    with pytest.raises(ValidationError, match="unique"):
        _state(newer, newer)


def test_metadata_rejects_more_than_twenty_sessions():
    newest = date(2026, 8, 28)
    sessions = tuple(
        FrozenBreadthContributorSession(
            date=newest - timedelta(days=offset),
            contributors=(_contributor(),),
        )
        for offset in range(21)
    )

    with pytest.raises(ValidationError):
        _state(*sessions)


def test_metadata_contributors_must_be_sorted_and_unique():
    with pytest.raises(ValidationError, match="sorted"):
        FrozenBreadthContributorSession(
            date=date(2026, 8, 28),
            contributors=(_contributor("BBB"), _contributor("AAA")),
        )
    with pytest.raises(ValidationError, match="unique"):
        FrozenBreadthContributorSession(
            date=date(2026, 8, 28),
            contributors=(_contributor("AAA"), _contributor("aaa")),
        )


def test_metadata_reader_rejects_wrong_market(tmp_path):
    path = tmp_path / "metadata.json.gz"
    write_static_breadth_contributor_metadata(
        path,
        _state(
            FrozenBreadthContributorSession(
                date=date(2026, 8, 28),
                contributors=(_contributor(),),
            )
        ),
    )

    with pytest.raises(StaticBreadthContributorMetadataBundleError, match="market"):
        read_static_breadth_contributor_metadata(path, expected_market="CA")


def test_metadata_reader_rejects_unsupported_schema(tmp_path):
    path = tmp_path / "metadata.json.gz"
    payload = {
        "schema_version": "legacy-v0",
        "market": "US",
        "generated_at": "2026-08-31T04:00:00Z",
        "sessions": [],
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(StaticBreadthContributorMetadataBundleError, match="schema"):
        read_static_breadth_contributor_metadata(path, expected_market="US")


def test_metadata_reader_rejects_corrupt_gzip(tmp_path):
    path = tmp_path / "metadata.json.gz"
    path.write_bytes(b"not gzip")

    with pytest.raises(StaticBreadthContributorMetadataBundleError, match="read"):
        read_static_breadth_contributor_metadata(path, expected_market="US")
