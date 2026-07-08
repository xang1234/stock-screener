from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.scripts.validate_static_market_artifacts import (
    StaticMarketArtifactValidationError,
    validate_market_artifacts,
)


def _write_market_manifest(base: Path, artifact_name: str, market: str) -> None:
    market_dir = base / artifact_name / "markets" / market.lower()
    market_dir.mkdir(parents=True)
    (market_dir / "manifest.market.json").write_text(
        json.dumps({"market": market}),
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


def test_static_market_validator_rejects_failed_selected_market_fallback(tmp_path: Path) -> None:
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

    with pytest.raises(StaticMarketArtifactValidationError) as exc_info:
        validate_market_artifacts(
            current_dir=current_dir,
            fallback_dir=fallback_dir,
            selected_markets={"CN"},
            expected_markets={"US", "CN"},
        )

    assert "Selected markets missing current artifacts: CN" in str(exc_info.value)


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
    assert result.statuses["CN"].reason == "not_trading_day"


def test_static_market_validator_rejects_selected_market_fallback_without_status(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    _write_market_manifest(current_dir, "static-market-US", "US")
    _write_market_manifest(fallback_dir, "static-market-CN", "CN")

    with pytest.raises(StaticMarketArtifactValidationError) as exc_info:
        validate_market_artifacts(
            current_dir=current_dir,
            fallback_dir=fallback_dir,
            selected_markets={"CN"},
            expected_markets={"US", "CN"},
        )

    assert "Selected markets missing current artifacts: CN" in str(exc_info.value)
    assert "missing status artifact" in str(exc_info.value)


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
