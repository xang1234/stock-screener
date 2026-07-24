"""Validate static-site market artifacts before publishing."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.domain.markets import market_registry
from app.services.static_market_artifact_contract import (
    STATIC_MARKET_METADATA_FILENAME,
    StaticMarketArtifactContractError,
    expected_market_from_static_market_manifest_path,
    read_static_market_manifest,
)
from app.services.static_market_publish_policy import OPTIONAL_STATIC_MARKETS


class StaticMarketArtifactValidationError(RuntimeError):
    """Raised when static market artifacts are not safe to publish."""


_VALID_STATUS_VALUES = frozenset({"published", "skipped", "failed"})
_VALID_REASON_VALUES = frozenset(
    {
        "not_trading_day",
        "no_current_artifact",
        "export_failed",
    }
)
_ALLOWED_MISSING_MARKETS = OPTIONAL_STATIC_MARKETS
_ALLOWED_MISSING_REASONS = frozenset({"not_trading_day", "no_current_artifact"})


@dataclass(frozen=True)
class MarketArtifactStatus:
    """Canonical per-market outcome from the current build-market job."""

    market: str
    has_current_artifact: bool
    status: str
    reason: str | None

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        source: Path | None = None,
    ) -> "MarketArtifactStatus":
        label = str(source) if source is not None else "status payload"
        if not isinstance(payload, dict):
            raise StaticMarketArtifactValidationError(f"{label}: status payload must be an object.")

        market = str(payload.get("market", "")).strip().upper()
        if not market:
            raise StaticMarketArtifactValidationError(f"{label}: market is required.")

        has_current_artifact = payload.get("has_current_artifact")
        if not isinstance(has_current_artifact, bool):
            raise StaticMarketArtifactValidationError(
                f"{label}: has_current_artifact must be a boolean."
            )

        status = payload.get("status")
        if not isinstance(status, str):
            raise StaticMarketArtifactValidationError(f"{label}: status must be a string.")
        normalized_status = status.strip().lower()
        if normalized_status not in _VALID_STATUS_VALUES:
            valid = ", ".join(sorted(_VALID_STATUS_VALUES))
            raise StaticMarketArtifactValidationError(
                f"{label}: status must be one of: {valid}."
            )

        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise StaticMarketArtifactValidationError(f"{label}: reason must be a string or null.")
        normalized_reason = reason.strip().lower() if isinstance(reason, str) else None
        if normalized_reason is not None and normalized_reason not in _VALID_REASON_VALUES:
            valid = ", ".join(sorted(_VALID_REASON_VALUES))
            raise StaticMarketArtifactValidationError(
                f"{label}: reason must be one of: {valid}; or null."
            )

        return cls(
            market=market,
            has_current_artifact=has_current_artifact,
            status=normalized_status,
            reason=normalized_reason,
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
    allowed_missing_markets: set[str]

    @property
    def present_markets(self) -> set[str]:
        return self.current_markets | self.fallback_markets

    @property
    def selected_fallback_markets(self) -> set[str]:
        return (self.selected_markets & self.fallback_markets) - self.current_markets

    @property
    def selected_fallback_diagnostics(self) -> dict[str, str]:
        diagnostics: dict[str, str] = {}
        for market in self.selected_fallback_markets:
            status = self.statuses.get(market)
            diagnostics[market] = (
                status.diagnostic_label()
                if status is not None
                else "missing status artifact"
            )
        return diagnostics


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
    for manifest in base.rglob(STATIC_MARKET_METADATA_FILENAME):
        try:
            expected_market = expected_market_from_static_market_manifest_path(
                base,
                manifest,
            )
            payload = read_static_market_manifest(
                manifest,
                expected_market=expected_market,
            )
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        except StaticMarketArtifactContractError as exc:
            raise StaticMarketArtifactValidationError(
                str(exc)
            ) from exc
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
                json.loads(status_path.read_text(encoding="utf-8")),
                source=status_path,
            )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise StaticMarketArtifactValidationError(
                f"{status_path}: could not read status artifact ({exc})."
            ) from exc
        statuses[status.market] = status
    return statuses


def _normalize_markets(markets: Iterable[str]) -> set[str]:
    return {str(market).strip().upper() for market in markets if str(market).strip()}


def _can_tolerate_missing_current_artifact(
    status: MarketArtifactStatus | None,
) -> bool:
    if status is None:
        return True
    if status.has_current_artifact:
        return False
    return status.reason in _ALLOWED_MISSING_REASONS


def validate_market_artifacts(
    *,
    current_dir: Path,
    fallback_dir: Path,
    selected_markets: Iterable[str],
    expected_markets: Iterable[str] | None = None,
) -> StaticMarketArtifactValidationResult:
    """Require complete coverage while allowing last-known-good market fallbacks."""
    expected = (
        _normalize_markets(expected_markets)
        if expected_markets is not None
        else {code.upper() for code in market_registry.supported_market_codes()}
    )
    current_markets = collect_markets(current_dir)
    fallback_markets = collect_markets(fallback_dir)
    statuses = collect_statuses(current_dir)
    normalized_selected = _normalize_markets(selected_markets)
    missing_set = expected - (current_markets | fallback_markets)
    fallback_backed_omissions = (expected - current_markets) & fallback_markets
    allowed_missing = {
        market
        for market in missing_set & _ALLOWED_MISSING_MARKETS
        if _can_tolerate_missing_current_artifact(statuses.get(market))
    }
    disallowed_missing = sorted(missing_set - allowed_missing)
    fallback_status_checked_markets = _ALLOWED_MISSING_MARKETS | normalized_selected
    disallowed_fallback_omissions = sorted(
        market
        for market in fallback_backed_omissions & fallback_status_checked_markets
        if not _can_tolerate_missing_current_artifact(statuses.get(market))
    )
    result = StaticMarketArtifactValidationResult(
        expected_markets=expected,
        selected_markets=normalized_selected,
        current_markets=current_markets,
        fallback_markets=fallback_markets,
        statuses=statuses,
        allowed_missing_markets=allowed_missing,
    )
    if disallowed_missing:
        raise StaticMarketArtifactValidationError(
            "Refusing to publish an incomplete static site. No current build "
            f"and no fallback artifact for: {', '.join(disallowed_missing)}."
        )
    if disallowed_fallback_omissions:
        details = "; ".join(
            f"{market}: {statuses[market].diagnostic_label()}"
            for market in disallowed_fallback_omissions
            if market in statuses
        )
        suffix = f" Details: {details}." if details else ""
        raise StaticMarketArtifactValidationError(
            "Refusing to publish fallback static market artifacts for "
            "ineligible current statuses: "
            f"{', '.join(disallowed_fallback_omissions)}.{suffix}"
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
    if result.selected_fallback_markets:
        details = "; ".join(
            f"{market}: {reason}"
            for market, reason in sorted(result.selected_fallback_diagnostics.items())
        )
        print(
            "::warning::Publishing last-known-good fallback artifacts for selected markets: "
            f"{_format_market_list(result.selected_fallback_markets)}. Details: {details}.",
            flush=True,
        )
    if result.allowed_missing_markets:
        print(
            "::warning::Publishing without optional market artifacts for: "
            f"{_format_market_list(result.allowed_missing_markets)}.",
            flush=True,
        )
        print("Required market artifacts present; static site is publishable.")
    else:
        print("All supported markets present; static site is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
