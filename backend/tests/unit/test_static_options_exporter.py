from __future__ import annotations

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from app.services.static_options_artifact_selector import StaticOptionsArtifactSelector
from app.services.static_options_contract import validate_static_options_artifact
from app.services.static_options_exporter import StaticOptionsExporter
from app.services.static_options_section import StaticOptionsSection

METRICS = (
    "max_pain",
    "net_gex",
    "gamma_flip",
    "call_wall",
    "put_wall",
    "atm_iv",
    "skew_25_delta",
    "realized_volatility",
    "vrp",
    "activity_intensity",
    "volume_oi_ratio",
    "near_spot_volume_concentration",
)

HISTORICAL_METRICS = (
    "iv_percentile",
    "iv_rank",
    "max_pain_change_5",
    "net_gex_change_5",
    "gamma_flip_change_5",
    "atm_iv_change_5",
    "skew_25_delta_change_5",
    "realized_volatility_change_5",
    "vrp_change_5",
    "activity_intensity_change_5",
)


def _item(symbol: str, *, kind: str = "current", rank: int = 1):
    values = {
        name: (190.0 if name == "max_pain" else None)
        for name in (*METRICS, *HISTORICAL_METRICS)
    }
    evidence = {
        name: {
            "available": values[name] is not None,
            "reason_codes": [] if values[name] is not None else ["building_history"],
            "evidence": {},
        }
        for name in (*METRICS, *HISTORICAL_METRICS)
    }
    evidence["quality"] = {
        "source_spot_price": 200.0,
        "provider_spot_price": 201.0,
        "spot_disagreement_ratio": 0.005,
        "latest_contract_trade_at": "2026-09-04T20:00:00+00:00",
        "days_to_expiration": 42,
        "normalized_call_count": 30,
        "normalized_put_count": 30,
        "distinct_strike_count": 30,
        "open_interest_coverage": 1.0,
        "iv_coverage": 0.95,
        "volume_coverage": 0.9,
        "two_sided_quote_coverage": 0.85,
    }
    return SimpleNamespace(
        security_symbol=symbol,
        candidate_kind=kind,
        candidate_rank=rank if kind == "current" else None,
        leader_rank=None,
        observation_state="available",
        core_valid=True,
        spot_price=200.0,
        expiration=date(2026, 10, 16),
        observation_at=datetime(2026, 9, 4, 21, 30, tzinfo=UTC),
        call_open_interest=1000,
        put_open_interest=800,
        call_volume=300,
        put_volume=250,
        activity_rank=rank if kind == "current" else None,
        short_history_observation_count=2,
        iv_history_observation_count=2,
        lifetime_observation_count=4,
        retry_count=0,
        assumptions_json={"gex_model": "dealer_proxy"},
        warnings_json=[],
        reasons_json=["building_history"],
        evidence_json=evidence,
        strike_points=[
            SimpleNamespace(
                strike=195.0,
                call_open_interest=500,
                put_open_interest=400,
                call_volume=100,
                put_volume=80,
                call_iv=0.31,
                put_iv=0.33,
                estimated_call_gex=1250.0,
                estimated_put_gex=-900.0,
            )
        ],
        **values,
    )


def _run(*items):
    return SimpleNamespace(
        id=7,
        schema_version="options-analytics-v1",
        calculation_version="options-analytics-v1",
        source_feature_run_id=33,
        as_of_date=date(2026, 9, 4),
        market="US",
        provider="yahoo",
        created_at=datetime(2026, 9, 4, 21, 0, tzinfo=UTC),
        published_at=datetime(2026, 9, 4, 21, 40, tzinfo=UTC),
        expected_count=len(items),
        current_count=sum(item.candidate_kind == "current" for item in items),
        continuity_count=sum(item.candidate_kind == "continuity" for item in items),
        completed_count=len(items),
        core_valid_current_count=sum(
            item.candidate_kind == "current" and item.core_valid for item in items
        ),
        failed_count=0,
        retried_count=0,
        coverage=1.0,
        warnings_json=[],
        assumptions_json={"risk_free_rate": 0.04},
        items=list(items),
    )


class _Queries:
    def __init__(self, run, *, history=()):
        self.run = run
        self.history = tuple(history)

    def get_published_command_center(self, market):
        assert market == "US"
        return self.run

    def get_published_symbol_detail(self, symbol, market):
        from app.use_cases.options_analytics.queries import PublishedOptionsSymbolDetail

        item = next(row for row in self.run.items if row.security_symbol == symbol)
        return PublishedOptionsSymbolDetail(
            run=self.run,
            item=item,
            history=self.history,
        )

    def is_stale(self, run, market):
        return False


def test_exporter_writes_complete_current_only_artifact_with_history_gaps(tmp_path):
    current = _item("AAPL")
    continuity = _item("OLD", kind="continuity")
    history = (
        SimpleNamespace(
            as_of_date=date(2026, 8, 31),
            observation_state="available",
            **{
                name: getattr(current, name) for name in (*METRICS, *HISTORICAL_METRICS)
            },
        ),
        SimpleNamespace(
            as_of_date=date(2026, 9, 4),
            observation_state="available",
            **{
                name: getattr(current, name) for name in (*METRICS, *HISTORICAL_METRICS)
            },
        ),
    )
    destination = tmp_path / "options"

    manifest = StaticOptionsExporter(
        _Queries(_run(current, continuity), history=history)
    ).export(
        destination,
        generated_at="2026-09-04T22:00:00Z",
    )

    assert manifest["published_run_id"] == 7
    assert set(manifest["symbols"]) == {"AAPL"}
    assert (destination / "manifest.json").is_file()
    assert (destination / "command-center.json").is_file()
    detail_path = tmp_path / manifest["symbols"]["AAPL"]["path"]
    detail = json.loads(detail_path.read_text())
    assert [point["as_of_date"] for point in detail["history"]] == [
        "2026-08-31",
        "2026-09-04",
    ]
    validated = validate_static_options_artifact(destination)
    assert validated["coverage"] == 1.0


def test_exporter_keeps_previous_directory_if_staged_write_fails(tmp_path):
    destination = tmp_path / "options"
    destination.mkdir()
    (destination / "sentinel.txt").write_text("last-good")
    queries = _Queries(_run(_item("AAPL")))
    queries.get_published_symbol_detail = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("detail failed")
    )

    with pytest.raises(RuntimeError, match="detail failed"):
        StaticOptionsExporter(queries).export(
            destination,
            generated_at="2026-09-04T22:00:00Z",
        )

    assert (destination / "sentinel.txt").read_text() == "last-good"


def test_exporter_does_not_delete_an_unrelated_fixed_name_sibling(tmp_path):
    destination = tmp_path / "options"
    destination.mkdir()
    (destination / "old.txt").write_text("old")
    unrelated = tmp_path / ".options-previous"
    unrelated.mkdir()
    (unrelated / "operator.txt").write_text("keep")

    StaticOptionsExporter(_Queries(_run(_item("AAPL")))).export(
        destination,
        generated_at="2026-09-04T22:00:00Z",
    )

    assert (unrelated / "operator.txt").read_text() == "keep"


def test_static_site_composition_advertises_options_only_after_selection(tmp_path):
    run = _run(_item("AAPL"))
    section = StaticOptionsSection(
        enabled=True,
        exporter_factory=lambda _db: StaticOptionsExporter(_Queries(run)),
        selector=StaticOptionsArtifactSelector(),
    )
    entry = {
        "feature_run_id": 33,
        "as_of_date": "2026-09-04",
        "features": {"scan": True},
        "pages": {},
        "assets": {},
    }
    result = section.compose_live(
        db=SimpleNamespace(),
        output_dir=tmp_path,
        generated_at="2026-09-04T22:00:00Z",
        equity_entry=entry,
        fallback_options_dir=None,
    )

    assert result.warnings == ()
    assert result.selected is True
    assert entry["features"]["options"] is True
    assert entry["pages"]["options"] == {"path": "options/manifest.json"}
    assert (
        validate_static_options_artifact(tmp_path / "options")["published_run_id"] == 7
    )


def test_server_static_export_supplies_existing_options_as_last_good_fallback(
    monkeypatch,
    tmp_path,
):
    from app.services import static_site_export_service as service_module
    from app.tasks import static_export_tasks as task_module

    target = tmp_path / "static-data"
    (target / "options").mkdir(parents=True)
    (target / "options" / "sentinel.txt").write_text("last-good")
    captured = {}

    class _Service:
        def export(self, output_dir, **kwargs):
            captured.update(kwargs)
            output_dir.mkdir(parents=True)
            (output_dir / "manifest.json").write_text("{}")
            return SimpleNamespace(
                manifest={"markets": {"US": {}}},
                as_of_date="2026-09-04",
                warnings=(),
            )

    monkeypatch.setattr(
        service_module,
        "StaticSiteExportService",
        lambda _session_factory: _Service(),
    )

    result = task_module.export_static_site_data.run(output_dir=str(target))

    assert result["status"] == "completed"
    assert captured["options_fallback_dir"] == target / "options"
