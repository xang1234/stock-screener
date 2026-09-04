from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime

import pytest
from app.services.options_history_transfer import (
    OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION,
    OptionsHistoryTransfer,
    OptionsHistoryTransferError,
)


def _observation(*, symbol="AAPL", as_of_date="2026-09-04"):
    return {
        "external_source_feature_run_key": "US:2026-09-04:source-33",
        "as_of_date": as_of_date,
        "schema_version": "options-analytics-v1",
        "provider": "yahoo",
        "published_at": "2026-09-04T22:00:00Z",
        "risk_free_rate": 0.04,
        "run_assumptions": {"risk_free_source": "^IRX"},
        "symbol": symbol,
        "candidate_kind": "current",
        "candidate_rank": 1,
        "leader_rank": None,
        "spot_price": 200.0,
        "expiration": "2026-10-16",
        "observation_state": "available",
        "core_valid": True,
        "observation_at": "2026-09-04T21:30:00Z",
        "max_pain": 190.0,
        "net_gex": 120000.0,
        "gamma_flip": 198.0,
        "call_wall": 210.0,
        "put_wall": 185.0,
        "atm_iv": 0.32,
        "skew_25_delta": -0.02,
        "realized_volatility": 0.24,
        "vrp": 0.08,
        "activity_intensity": 0.75,
        "activity_rank": 1,
        "call_open_interest": 1000,
        "put_open_interest": 800,
        "call_volume": 300,
        "put_volume": 250,
        "volume_oi_ratio": 0.305,
        "near_spot_volume_concentration": 0.6,
        "short_history_observation_count": 5,
        "iv_history_observation_count": 5,
        "lifetime_observation_count": 8,
        "retry_count": 0,
        "evidence": {},
        "assumptions": {},
        "warnings": [],
        "reason_codes": [],
    }


class _Repository:
    def __init__(self):
        self.imported = set()

    def export_history_observations(self, market, calculation_version):
        assert (market, calculation_version) == ("US", "options-analytics-v1")
        return (_observation(),)

    def import_history_transfer(self, observations, **identity):
        added = 0
        for row in observations:
            key = (row.external_source_feature_run_key, row.symbol)
            if key not in self.imported:
                self.imported.add(key)
                added += 1
        return {"imported_observations": added, "identity": identity}


def _bundle():
    return OptionsHistoryTransfer(_Repository()).export_bundle(
        exported_at=datetime(2026, 9, 5, 1, 0, tzinfo=UTC)
    )


def test_transfer_bundle_is_checksummed_and_contains_aggregate_history_only():
    bundle = _bundle()

    assert bundle["schema_version"] == OPTIONS_HISTORY_TRANSFER_SCHEMA_VERSION
    assert bundle["calculation_version"] == "options-analytics-v1"
    assert bundle["market"] == "US"
    assert len(bundle["payload_checksum"]) == 64
    assert "last_current_memberships" not in bundle
    encoded = repr(bundle)
    assert "strike_points" not in encoded
    assert "raw_contract" not in encoded


def test_import_is_idempotent_through_repository():
    repository = _Repository()
    transfer = OptionsHistoryTransfer(repository)
    bundle = transfer.export_bundle(exported_at=datetime(2026, 9, 5, 1, 0, tzinfo=UTC))

    first = transfer.import_bundle(bundle, today=date(2026, 9, 5))
    second = transfer.import_bundle(bundle, today=date(2026, 9, 5))

    assert first["imported_observations"] == 1
    assert second["imported_observations"] == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(payload_checksum="0" * 64), "checksum"),
        (lambda value: value.update(market="HK"), "market"),
        (lambda value: value.update(schema_version="future-v9"), "schema"),
        (lambda value: value.update(calculation_version="future-v9"), "calculation"),
        (
            lambda value: value["observations"].append(
                deepcopy(value["observations"][0])
            ),
            "duplicate",
        ),
        (
            lambda value: value["observations"][0].update(atm_iv=float("nan")),
            "non-finite",
        ),
        (
            lambda value: value["observations"][0].update(as_of_date="2026-09-06"),
            "future",
        ),
    ],
)
def test_transfer_rejects_invalid_or_incompatible_payloads(mutate, message):
    bundle = _bundle()
    mutate(bundle)

    with pytest.raises(OptionsHistoryTransferError, match=message):
        OptionsHistoryTransfer(_Repository()).import_bundle(
            bundle,
            today=date(2026, 9, 5),
        )
