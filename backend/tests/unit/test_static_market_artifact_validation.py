from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.scripts.validate_static_market_artifacts import (
    StaticMarketArtifactValidationError,
    main,
    validate_market_artifacts,
)


def _write_market_manifest(
    base: Path,
    artifact_name: str,
    market: str,
    *,
    schema_version: str = "static-site-v3",
) -> None:
    market_dir = base / artifact_name / "markets" / market.lower()
    market_dir.mkdir(parents=True)
    (market_dir / "manifest.market.json").write_text(
        json.dumps({"market": market, "schema_version": schema_version}),
        encoding="utf-8",
    )


def _write_market_status(
    base: Path,
    market: str,
    *,
    has_current_artifact: bool,
    status: str,
    reason: str | None,
) -> None:
    status_dir = base / f"static-market-status-{market}"
    status_dir.mkdir(parents=True)
    (status_dir / "status.json").write_text(
        json.dumps(
            {
                "market": market,
                "has_current_artifact": has_current_artifact,
                "status": status,
                "reason": reason,
            }
        ),
        encoding="utf-8",
    )


def _write_raw_market_status(base: Path, market: str, payload: object) -> None:
    status_dir = base / f"static-market-status-{market}"
    status_dir.mkdir(parents=True)
    (status_dir / "status.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_static_market_validator_allows_failed_selected_market_fallback(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-CN", "CN")
    _write_market_status(
        current_dir,
        "CN",
        has_current_artifact=False,
        status="failed",
        reason="no_current_artifact",
    )

    result = validate_market_artifacts(
        current_dir=current_dir,
        fallback_dir=fallback_dir,
        selected_markets={"CN"},
        expected_markets={"US", "CN"},
    )

    assert result.current_markets == {"US"}
    assert result.fallback_markets == {"CN"}
    assert result.selected_fallback_markets == {"CN"}
    assert result.selected_fallback_diagnostics == {
        "CN": "status failed/no_current_artifact"
    }


def test_static_market_validator_allows_selected_market_fallback_for_non_trading_day(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-CN", "CN")
    _write_market_status(
        current_dir,
        "CN",
        has_current_artifact=False,
        status="skipped",
        reason="not_trading_day",
    )

    result = validate_market_artifacts(
        current_dir=current_dir,
        fallback_dir=fallback_dir,
        selected_markets={"CN"},
        expected_markets={"US", "CN"},
    )

    assert result.current_markets == {"US"}
    assert result.fallback_markets == {"CN"}
    assert result.selected_fallback_markets == {"CN"}
    assert result.statuses["CN"].reason == "not_trading_day"


def test_static_market_validator_allows_selected_market_fallback_without_status(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-CN", "CN")

    result = validate_market_artifacts(
        current_dir=current_dir,
        fallback_dir=fallback_dir,
        selected_markets={"CN"},
        expected_markets={"US", "CN"},
    )

    assert result.selected_fallback_markets == {"CN"}
    assert result.selected_fallback_diagnostics == {"CN": "missing status artifact"}


def test_static_market_validator_warns_when_selected_market_uses_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-CN", "CN")
    _write_market_status(
        current_dir,
        "CN",
        has_current_artifact=False,
        status="failed",
        reason="no_current_artifact",
    )
    monkeypatch.setattr(
        "app.scripts.validate_static_market_artifacts.market_registry.supported_market_codes",
        lambda: ("US", "CN"),
    )

    exit_code = main(
        [
            "--current-dir",
            str(current_dir),
            "--fallback-dir",
            str(fallback_dir),
            "--selected-markets",
            '["CN"]',
        ]
    )

    assert exit_code == 0
    assert (
        "::warning::Publishing last-known-good fallback artifacts for selected markets: CN. "
        "Details: CN: status failed/no_current_artifact."
    ) in capsys.readouterr().out


def test_static_market_validator_rejects_string_boolean_status_contract(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-CN", "CN")
    _write_raw_market_status(
        current_dir,
        "CN",
        {
            "market": "CN",
            "has_current_artifact": "false",
            "status": "skipped",
            "reason": "not_trading_day",
        },
    )

    with pytest.raises(StaticMarketArtifactValidationError) as exc_info:
        validate_market_artifacts(
            current_dir=current_dir,
            fallback_dir=fallback_dir,
            selected_markets={"CN"},
            expected_markets={"US", "CN"},
        )

    assert "static-market-status-CN/status.json" in str(exc_info.value)
    assert "has_current_artifact must be a boolean" in str(exc_info.value)


def test_static_market_validator_rejects_unknown_status_values(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-CN", "CN")
    _write_raw_market_status(
        current_dir,
        "CN",
        {
            "market": "CN",
            "has_current_artifact": False,
            "status": "maybe",
            "reason": "not_trading_day",
        },
    )

    with pytest.raises(StaticMarketArtifactValidationError) as exc_info:
        validate_market_artifacts(
            current_dir=current_dir,
            fallback_dir=fallback_dir,
            selected_markets={"CN"},
            expected_markets={"US", "CN"},
        )

    assert "status must be one of" in str(exc_info.value)


def test_static_market_validator_rejects_incompatible_fallback_schema(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(
        fallback_dir,
        "static-market-AU",
        "AU",
        schema_version="static-site-v2",
    )

    with pytest.raises(StaticMarketArtifactValidationError) as exc_info:
        validate_market_artifacts(
            current_dir=current_dir,
            fallback_dir=fallback_dir,
            selected_markets={"US"},
            expected_markets={"US", "AU"},
        )

    assert "static-market-AU" in str(exc_info.value)
    assert "schema_version 'static-site-v2'; expected 'static-site-v3'" in str(
        exc_info.value
    )


def test_static_market_validator_rejects_missing_manifest_market(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    market_dir = fallback_dir / "static-market-AU" / "markets" / "au"
    market_dir.mkdir(parents=True)
    (market_dir / "manifest.market.json").write_text(
        json.dumps({"schema_version": "static-site-v3"}),
        encoding="utf-8",
    )

    with pytest.raises(StaticMarketArtifactValidationError) as exc_info:
        validate_market_artifacts(
            current_dir=current_dir,
            fallback_dir=fallback_dir,
            selected_markets={"US"},
            expected_markets={"US", "AU"},
        )

    assert "static-market-AU" in str(exc_info.value)
    assert "market is required" in str(exc_info.value)


def test_static_market_validator_rejects_swapped_manifest_market(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-AU", "US")

    with pytest.raises(StaticMarketArtifactValidationError) as exc_info:
        validate_market_artifacts(
            current_dir=current_dir,
            fallback_dir=fallback_dir,
            selected_markets={"US"},
            expected_markets={"US", "AU"},
        )

    assert "static-market-AU" in str(exc_info.value)
    assert "market 'US'; expected 'AU'" in str(exc_info.value)
