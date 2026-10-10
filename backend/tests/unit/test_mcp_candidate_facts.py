"""Regression tests: the MCP candidate record exposes the persisted facts.

The published feature run already carries the strategy ratings, the Setup
Engine facts and the VCP detail block. ``_candidate_record`` projected only a
small slice of them, so an external client could not see them at all.

Each test drives the tools through the public ``call_tool`` entry point where a
tool is involved, so it fails on a projection regression rather than on a
helper changing shape. Nothing here asserts a recomputed value: every expected
number equals what the feature run was seeded with.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.infra.db.repositories.feature_store_repo import _vcp_num_bases_from_details
from app.interfaces.mcp.market_copilot import MarketCopilotService
from tests.helpers.mcp_fixture import (
    create_mcp_test_session_factory,
    seed_market_copilot_data,
)

# The keys ``_candidate_record`` returned before this change. Their names and
# values are the compatibility surface for existing clients, so they are
# pinned as a whole dict instead of one assertion per key.
PRE_EXISTING_KEYS = {
    "symbol": "NVDA",
    "company_name": "NVIDIA Corporation",
    "composite_score": 92.0,
    "rating": "Strong Buy",
    "current_price": 145.0,
    "rs_rating": 95,
    "stage": 2,
    "gics_sector": "Information Technology",
    "ibd_industry_group": "Semiconductors",
    "market_cap": 3_000_000_000_000,
    "volume": 26_000_000,
    "eps_growth_qq": 36.0,
    "sales_growth_qq": 30.0,
    "se_setup_score": 87.0,
    "se_setup_ready": True,
}


@pytest.fixture()
def session_factory():
    """In-memory database carrying the shared MCP seed data."""
    factory, _engine = create_mcp_test_session_factory()
    seed_market_copilot_data(factory)
    return factory


@pytest.fixture()
def service(session_factory):
    """Read-only copilot service over the seeded session factory."""
    return MarketCopilotService(
        session_factory,
        SimpleNamespace(
            mcp_watchlist_writes_enabled=False,
            mcp_server_name="stockscreen-market-copilot",
        ),
    )


def _payload(service, tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool and return the envelope, asserting it did not error."""
    result = service.call_tool(tool_name, arguments)
    assert result.get("isError") is not True
    return result["structuredContent"]


def _explain_result(service, symbol: str, depth: str = "full") -> dict:
    """Return the ``result`` payload of an ``explain_symbol`` call."""
    return _payload(service, "explain_symbol", {"symbol": symbol, "depth": depth})["result"]


def _candidate_record(service, run_id: int, symbol: str) -> dict:
    """Return the projected record straight from the repository mapping."""
    with service._uow_scope() as uow:
        item = uow.feature_store.get_by_symbol_for_run(
            run_id,
            symbol,
            include_sparklines=False,
            include_setup_payload=False,
        )
    assert item is not None
    return service._candidate_record(item)


def test_candidate_record_keeps_its_pre_existing_keys_and_values(service):
    """Compatibility: every key that existed before still carries its value."""
    record = _candidate_record(service, 2, "NVDA")

    for key, expected in PRE_EXISTING_KEYS.items():
        assert record[key] == expected, key


def test_candidate_record_projects_the_persisted_screener_ratings(session_factory, service):
    """The three strategy ratings are persisted per row and must pass through."""
    with session_factory() as db:
        row = (
            db.query(FeatureRun)
            .filter(FeatureRun.id == 2)
            .one()
        )
        assert row.status == "published"

    record = _candidate_record(service, 2, "NVDA")

    # Seeded values: rating "Strong Buy" -> minervini "Strong Buy",
    # canslim "Buy" (downgraded for a strong buy), volume breakthrough
    # "Strong Buy" above the 90 composite.
    assert record["minervini_rating"] == "Strong Buy"
    assert record["canslim_rating"] == "Buy"
    assert record["volume_breakthrough_rating"] == "Strong Buy"


def test_candidate_record_projects_the_persisted_row_state(service):
    """Coverage and opportunity facts are stored per row, not derived."""
    record = _candidate_record(service, 2, "NVDA")

    assert record["data_status"] == "ok"
    assert record["is_scannable"] is True
    assert record["action_state"] == "WATCH"
    assert record["opportunity_state"]["policy_version"] == "correction-survivors-v1"
    assert record["opportunity_state"]["metrics"]["hard_invalidation"] is False
    assert record["opportunity_state"]["data_availability"]["required_evidence"] == "complete"


def test_candidate_record_projects_the_persisted_canonical_setup_facts(service):
    """The canonical Setup Engine fields, unchanged, including the pivot gap."""
    record = _candidate_record(service, 2, "NVDA")

    assert record["se_setup_score"] == 87.0
    assert record["se_setup_ready"] is True
    assert record["se_readiness_score"] == 89.0
    assert record["se_quality_score"] == 82.0
    assert record["se_pattern_primary"] == "VCP"
    assert record["se_pivot_price"] == 150.8
    assert record["se_pivot_type"] == "cup_with_handle"
    assert record["se_pivot_date"] == "2026-03-27"
    assert record["se_distance_to_pivot_pct"] == 4.0
    assert record["se_in_early_zone"] is False
    assert record["se_extended_from_pivot"] is False
    assert record["se_atr14_pct"] == 3.5
    assert record["se_volume_vs_50d"] == 1.4


def test_candidate_record_projects_vcp_num_bases_when_persisted(service):
    """A stored Minervini VCP block yields its base count."""
    record = _candidate_record(service, 2, "NVDA")

    assert record["vcp_detected"] is True
    assert record["vcp_num_bases"] == 3


def test_candidate_record_reports_none_when_no_base_count_was_persisted(service):
    """No stored block, no count — and no derivation from ``vcp_detected``.

    ``MSFT`` in run 2 is seeded with ``vcp_detected=True`` but without a
    ``full_analysis.vcp`` block. Deriving the count from the flag, or from the
    length of the candidate list, would produce a number here.
    """
    record = _candidate_record(service, 2, "MSFT")

    assert record["vcp_detected"] is True
    assert record["vcp_num_bases"] is None


def test_candidate_record_projects_volume_surge(service):
    """Volume surge is a persisted fact, not a threshold applied on read."""
    high_volume = _candidate_record(service, 2, "NVDA")
    low_volume = _candidate_record(service, 2, "SNOW")

    assert high_volume["volume_surge"] is True
    assert low_volume["volume_surge"] is False


def test_explain_symbol_full_surfaces_the_facts_on_the_result(service):
    """``explain_symbol`` must expose them under ``result``.

    The run is resolved symbol/market-wise, so the assertion holds without any
    global ``latest_published`` pointer: the per-market pointer is what the
    tool uses. The full depth is what carries the candidate record.
    """
    result = _explain_result(service, "NVDA")

    assert result["minervini_rating"] == "Strong Buy"
    assert result["canslim_rating"] == "Buy"
    assert result["volume_breakthrough_rating"] == "Strong Buy"
    assert result["vcp_detected"] is True
    assert result["vcp_num_bases"] == 3
    assert result["se_distance_to_pivot_pct"] == 4.0
    assert result["se_pivot_price"] == 150.8
    assert result["se_readiness_score"] == 89.0
    assert result["action_state"] == "WATCH"


def test_explain_symbol_full_works_without_the_global_pointer(service, session_factory):
    """No regression of the market-pointer fix: the global pointer is not needed.

    ``_point`` mirrors what the publish path writes; the global pointer is
    removed here, and the per-market pointer keeps ``explain_symbol`` working.
    """
    with session_factory() as db:
        db.query(FeatureRunPointer).filter(FeatureRunPointer.key == "latest_published").delete()
        db.add(FeatureRunPointer(key="latest_published_market:US", run_id=2))
        db.commit()

    result = _explain_result(service, "NVDA")

    assert result["vcp_num_bases"] == 3
    assert result["se_setup_score"] == 87.0


VCP_INCOMPLETE_DETAILS = [
    pytest.param({}, id="empty-details"),
    pytest.param({"details": None}, id="details-not-a-mapping"),
    pytest.param({"details": {"screeners": {}}}, id="no-minervini-screener"),
    pytest.param({"details": {"screeners": {"minervini": {"details": {}}}}}, id="no-full-analysis"),
    pytest.param(
        {"details": {"screeners": {"minervini": {"details": {"full_analysis": {"vcp": {}}}}}}},
        id="no-num-bases",
    ),
    pytest.param(
        {
            "details": {
                "screeners": {
                    "minervini": {
                        "details": {"full_analysis": {"vcp": {"num_bases": None}}}
                    }
                }
            }
        },
        id="num-bases-null",
    ),
    pytest.param(
        {
            "details": {
                "screeners": {
                    "minervini": {
                        "details": {"full_analysis": {"vcp": {"num_bases": "5"}}}
                    }
                }
            }
        },
        id="num-bases-string",
    ),
    pytest.param(
        {
            "details": {
                "screeners": {
                    "minervini": {
                        "details": {"full_analysis": {"vcp": {"num_bases": True}}}
                    }
                }
            }
        },
        id="num-bases-bool",
    ),
    pytest.param({"details": {"screeners": ["minervini"]}}, id="screeners-not-a-mapping"),
]


@pytest.mark.parametrize("details", VCP_INCOMPLETE_DETAILS)
def test_vcp_num_bases_is_none_when_the_stored_path_is_incomplete(details):
    """Missing nodes, non-mappings and non-integral values all yield ``None``.

    The boolean case matters most: ``bool`` is an ``int`` subclass, so a plain
    ``isinstance(value, int)`` check would report ``True`` as a base count.
    """
    assert _vcp_num_bases_from_details(details) is None


def _vcp_details(num_bases):
    """Build the persisted details blob the VCP base count is read from."""
    return {
        "details": {
            "screeners": {
                "minervini": {"details": {"full_analysis": {"vcp": {"num_bases": num_bases}}}}
            }
        }
    }


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(3, 3, id="plain-int"),
        pytest.param(0, 0, id="zero-is-a-value"),
        pytest.param(5.0, 5, id="integral-float"),
    ],
)
def test_vcp_num_bases_reads_the_stored_integer(value, expected):
    """Stored counts pass through unchanged, including an integral float."""
    assert _vcp_num_bases_from_details(_vcp_details(value)) == expected


def test_vcp_num_bases_rejects_a_fractional_count():
    """A fractional float is not a base count and must not be truncated."""
    assert _vcp_num_bases_from_details(_vcp_details(2.5)) is None


@pytest.mark.parametrize("value", [pytest.param(-1, id="int"), pytest.param(-1.0, id="float")])
def test_vcp_num_bases_rejects_a_negative_count(value):
    """A count is never negative; both numeric branches must reject it.

    A negative stored value would otherwise reach ``extended_fields`` on both
    scan mappers and be handed to MCP clients as a base count.
    """
    assert _vcp_num_bases_from_details(_vcp_details(value)) is None
