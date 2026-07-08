"""Validate static-site market artifacts before publishing."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.domain.markets import market_registry


class StaticMarketArtifactValidationError(RuntimeError):
    """Raised when static market artifacts are not safe to publish."""


@dataclass(frozen=True)
class MarketArtifactStatus:
    """Canonical per-market outcome from the current build-market job."""

    market: str
    has_current_artifact: bool
    status: str
    reason: str | None

    @classmethod
    def from_payload(cls, payload: object) -> "MarketArtifactStatus | None":
        if not isinstance(payload, dict):
            return None
        market = str(payload.get("market", "")).strip().upper()
        if not market:
            return None
        reason = payload.get("reason")
        return cls(
            market=market,
            has_current_artifact=bool(payload.get("has_current_artifact", False)),
            status=str(payload.get("status", "")).strip().lower(),
            reason=str(reason).strip().lower() if reason is not None else None,
        )

    def allows_selected_fallback(self) -> bool:
        return (
            not self.has_current_artifact
            and self.status == "skipped"
            and self.reason == "not_trading_day"
        )

    def diagnostic_label(self) -> str:
        if self.reason:
            return f"status {self.status}/{self.reason}"
        return f"status {self.status or '(missing)'}"


@dataclass(frozen=True)
class StaticMarketArtifactValidationResult:
    expected_markets: set[str]
    selected_markets: set[str]
    current_markets: set[str]
    fallback_markets: set[str]
    statuses: dict[str, MarketArtifactStatus]

    @property
    def present_markets(self) -> set[str]:
        return self.current_markets | self.fallback_markets


def parse_selected_markets(raw: str | None) -> set[str]:
    if not raw:
        return set()
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise StaticMarketArtifactValidationError("SELECTED_MARKETS is not a JSON array.")
    return {str(market).strip().upper() for market in payload if str(market).strip()}


def collect_markets(base: Path) -> set[str]:
    markets: set[str] = set()
    if not base.exists():
        return markets
    for manifest in base.rglob("manifest.market.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        market = str(payload.get("market", "")).strip().upper()
        if market:
            markets.add(market)
    return markets


def collect_statuses(base: Path) -> dict[str, MarketArtifactStatus]:
    statuses: dict[str, MarketArtifactStatus] = {}
    if not base.exists():
        return statuses
    for status_path in base.rglob("status.json"):
        try:
            status = MarketArtifactStatus.from_payload(
                json.loads(status_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if status is not None:
            statuses[status.market] = status
    return statuses


def _normalize_markets(markets: Iterable[str]) -> set[str]:
    return {str(market).strip().upper() for market in markets if str(market).strip()}


def _selected_markets_missing_current_artifacts(
    *,
    selected_markets: set[str],
    current_markets: set[str],
    statuses: dict[str, MarketArtifactStatus],
) -> dict[str, str]:
    missing: dict[str, str] = {}
    for market in selected_markets:
        if market in current_markets:
            continue
        status = statuses.get(market)
        if status is not None and status.allows_selected_fallback():
            continue
        missing[market] = status.diagnostic_label() if status is not None else "missing status artifact"
    return missing


def validate_market_artifacts(
    *,
    current_dir: Path,
    fallback_dir: Path,
    selected_markets: Iterable[str],
    expected_markets: Iterable[str] | None = None,
) -> StaticMarketArtifactValidationResult:
    expected = (
        _normalize_markets(expected_markets)
        if expected_markets is not None
        else {code.upper() for code in market_registry.supported_market_codes()}
    )
    result = StaticMarketArtifactValidationResult(
        expected_markets=expected,
        selected_markets=_normalize_markets(selected_markets),
        current_markets=collect_markets(current_dir),
        fallback_markets=collect_markets(fallback_dir),
        statuses=collect_statuses(current_dir),
    )

    missing = sorted(result.expected_markets - result.present_markets)
    if missing:
        raise StaticMarketArtifactValidationError(
            "Refusing to publish an incomplete static site. No current build "
            f"and no fallback artifact for: {', '.join(missing)}."
        )

    selected_missing = _selected_markets_missing_current_artifacts(
        selected_markets=result.selected_markets,
        current_markets=result.current_markets,
        statuses=result.statuses,
    )
    if selected_missing:
        details = "; ".join(
            f"{market}: {reason}" for market, reason in sorted(selected_missing.items())
        )
        raise StaticMarketArtifactValidationError(
            "Refusing to publish stale fallback data for a selected market. "
            "Selected markets missing current artifacts: "
            f"{', '.join(sorted(selected_missing))}. Details: {details}."
        )

    return result


def _format_market_list(markets: Iterable[str]) -> str:
    values = sorted(markets)
    return ", ".join(values) if values else "(none)"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--fallback-dir", type=Path, required=True)
    parser.add_argument("--selected-markets", default="[]")
    args = parser.parse_args(argv)

    try:
        selected_markets = parse_selected_markets(args.selected_markets)
        result = validate_market_artifacts(
            current_dir=args.current_dir,
            fallback_dir=args.fallback_dir,
            selected_markets=selected_markets,
        )
    except (json.JSONDecodeError, StaticMarketArtifactValidationError) as exc:
        print(f"::error::{exc}", flush=True)
        return 1

    print(f"Expected markets: {_format_market_list(result.expected_markets)}")
    print(f"Present markets:  {_format_market_list(result.present_markets)}")
    if result.selected_markets:
        print(f"Selected markets: {_format_market_list(result.selected_markets)}")
    print("All supported markets present; static site is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
