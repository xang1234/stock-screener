"""Unit tests for asia.8.7 backend plumbing.

Verifies data-availability transparency flows end-to-end through the scan
result pipeline:

1. ``_unpack_joined_row`` merges ownership/sentiment + growth-cadence
   availability into a single ``field_availability`` dict, filtered to
   non-available entries only.
2. ``ScanResultItem.from_domain`` surfaces ``field_availability``,
   ``growth_reporting_cadence``, and ``growth_metric_basis`` on the HTTP
   response.
3. Feature-store path (``_unpack_feature_joined_row``) exposes the same
   signal.
"""

from __future__ import annotations

from app.domain.scanning.models import ScanResultItemDomain
from app.infra.db.repositories.feature_store_repo import _unpack_feature_joined_row
from app.infra.db.repositories.scan_result_repo import _unpack_joined_row
from app.schemas.scanning import ScanResultItem
from app.services.field_capability_registry import (
    REASON_CODE_NON_US_GAP,
    REASON_CODE_POLICY_EXCLUDED,
)
from app.services.growth_cadence_service import (
    BASIS_COMPARABLE_YOY,
    BASIS_QOQ,
    BASIS_UNAVAILABLE,
    CADENCE_INSUFFICIENT,
    CADENCE_QUARTERLY,
    CADENCE_SEMIANNUAL,
    REASON_COMPARABLE_YOY_FALLBACK,
    REASON_INSUFFICIENT_HISTORY,
)


def _make_item(**extended: object) -> ScanResultItemDomain:
    return ScanResultItemDomain(
        symbol="0700.HK",
        composite_score=80.0,
        rating="Buy",
        current_price=410.0,
        screener_outputs={},
        screeners_run=["minervini"],
        composite_method="weighted_average",
        screeners_passed=1,
        screeners_total=1,
        extended_fields=extended,
    )


def _scan_row(
    *,
    market: str | None,
    institutional_ownership: float | None = None,
    insider_ownership: float | None = None,
    short_interest: int | None = None,
    growth_reporting_cadence: str | None = None,
    growth_metric_basis: str | None = None,
) -> tuple:
    """Build a synthetic _scan_results_query row tuple.

    Matches the SELECT order: ScanResult, name, market, exchange, currency,
    market_cap_usd, adv_usd, institutional_ownership, insider_ownership,
    short_interest, growth_reporting_cadence, growth_metric_basis.
    """
    # The first element is the ORM ScanResult row — for _unpack_joined_row
    # we only read columns via the joined dict, so a sentinel object works.
    return (
        object(),  # ScanResult
        "Tencent",  # StockUniverse.name
        market,  # StockUniverse.market
        "HKEX",  # StockUniverse.exchange
        "HKD",  # StockUniverse.currency
        500_000_000_000,  # StockFundamental.market_cap_usd
        12_500_000,  # StockFundamental.adv_usd
        institutional_ownership,
        insider_ownership,
        short_interest,
        growth_reporting_cadence,
        growth_metric_basis,
    )


def _feature_row(**kwargs) -> tuple:
    """Mirror _scan_row for _unpack_feature_joined_row (same column order)."""
    return _scan_row(**kwargs)


# ---------------------------------------------------------------------------
# Legacy scan_result path: _unpack_joined_row
# ---------------------------------------------------------------------------


class TestUnpackJoinedRowFieldAvailability:
    def test_hk_row_with_missing_ownership_emits_unsupported_entries(self):
        # HK market + all ownership values None. Because finviz is excluded
        # from the HK provider chain, the ownership helper reports
        # ``policy_excluded`` for each ownership/sentiment field — the
        # non-US-gap code only fires when the provider IS in the chain but
        # the data happens to be missing.
        row = _scan_row(market="HK")
        _, joined = _unpack_joined_row(row)

        fa = joined["field_availability"]
        assert fa is not None
        assert set(fa.keys()) >= {
            "institutional_ownership",
            "insider_ownership",
            "short_interest",
        }
        for field in ("institutional_ownership", "insider_ownership", "short_interest"):
            entry = fa[field]
            assert entry["status"] == "unsupported"
            # Reason is either policy-excluded or non-US gap depending on
            # the field's canonical provider; both are valid HK outcomes.
            assert entry["reason_code"] in (
                REASON_CODE_POLICY_EXCLUDED,
                REASON_CODE_NON_US_GAP,
            )

    def test_hk_row_with_unavailable_growth_basis_adds_growth_entries(self):
        row = _scan_row(
            market="HK",
            growth_metric_basis=BASIS_UNAVAILABLE,
            growth_reporting_cadence=CADENCE_INSUFFICIENT,
        )
        _, joined = _unpack_joined_row(row)

        fa = joined["field_availability"]
        assert "eps_growth_qq" in fa
        assert fa["eps_growth_qq"]["reason_code"] == REASON_INSUFFICIENT_HISTORY
        assert fa["sales_growth_qq"]["status"] == "unavailable"
        # Flat growth metadata passes through.
        assert joined["growth_metric_basis"] == BASIS_UNAVAILABLE
        assert joined["growth_reporting_cadence"] == CADENCE_INSUFFICIENT

    def test_jp_row_with_comparable_yoy_basis_flags_computed(self):
        row = _scan_row(
            market="JP",
            growth_metric_basis=BASIS_COMPARABLE_YOY,
            growth_reporting_cadence=CADENCE_SEMIANNUAL,
        )
        _, joined = _unpack_joined_row(row)

        fa = joined["field_availability"]
        # Growth fields flagged as computed (fallback from comparable-period YoY).
        assert fa["eps_growth_qq"]["status"] == "computed"
        assert fa["eps_growth_qq"]["reason_code"] == REASON_COMPARABLE_YOY_FALLBACK

    def test_us_row_with_full_ownership_data_emits_no_entries(self):
        # US + all ownership values present + quarterly QoQ → nothing to surface.
        row = _scan_row(
            market="US",
            institutional_ownership=0.85,
            insider_ownership=0.05,
            short_interest=1_000_000,
            growth_metric_basis=BASIS_QOQ,
            growth_reporting_cadence=CADENCE_QUARTERLY,
        )
        _, joined = _unpack_joined_row(row)

        # Merged dict is None (falsy) because every ownership entry was
        # available and growth basis was QoQ (no cadence entry).
        assert joined["field_availability"] is None

    def test_only_available_entries_are_filtered_out(self):
        # Partial US data: institutional set, insider/short missing.
        # Because the ownership helper reports MISSING_SUPPORTED for the gaps
        # on a US row (not UNSUPPORTED), those still get entries — the filter
        # only strips status == "available".
        row = _scan_row(
            market="US",
            institutional_ownership=0.7,  # present → available → filtered out
            insider_ownership=None,       # missing → kept
            short_interest=None,          # missing → kept
        )
        _, joined = _unpack_joined_row(row)

        fa = joined["field_availability"]
        assert fa is not None
        assert "institutional_ownership" not in fa
        assert "insider_ownership" in fa
        assert "short_interest" in fa


# ---------------------------------------------------------------------------
# Feature-store path: _unpack_feature_joined_row (same SELECT order)
# ---------------------------------------------------------------------------


class TestUnpackFeatureJoinedRowFieldAvailability:
    def test_feature_store_path_mirrors_legacy_path_for_hk(self):
        row = _feature_row(
            market="HK",
            growth_metric_basis=BASIS_UNAVAILABLE,
            growth_reporting_cadence=CADENCE_INSUFFICIENT,
        )
        _, joined = _unpack_feature_joined_row(row)

        fa = joined["field_availability"]
        assert fa is not None
        assert "eps_growth_qq" in fa
        assert fa["eps_growth_qq"]["reason_code"] == REASON_INSUFFICIENT_HISTORY
        # Ownership fields also flagged (non-US gap).
        assert "institutional_ownership" in fa


# ---------------------------------------------------------------------------
# Domain → HTTP schema mapping
# ---------------------------------------------------------------------------


class TestScanResultItemMapping:
    def test_field_availability_and_growth_metadata_surface_on_response(self):
        fa = {
            "eps_growth_qq": {
                "status": "unavailable",
                "reason_code": REASON_INSUFFICIENT_HISTORY,
                "support_state": "unsupported",
            }
        }
        item = _make_item(
            field_availability=fa,
            growth_reporting_cadence=CADENCE_INSUFFICIENT,
            growth_metric_basis=BASIS_UNAVAILABLE,
        )
        resp = ScanResultItem.from_domain(item)
        assert resp.field_availability == fa
        assert resp.growth_reporting_cadence == CADENCE_INSUFFICIENT
        assert resp.growth_metric_basis == BASIS_UNAVAILABLE

    def test_missing_transparency_fields_default_to_none(self):
        item = _make_item()  # no transparency extended_fields
        resp = ScanResultItem.from_domain(item)
        assert resp.field_availability is None
        assert resp.growth_reporting_cadence is None
        assert resp.growth_metric_basis is None
