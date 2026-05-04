"""Market scan gating rules backed by runtime activity state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..database import SessionLocal
from ..domain.markets import Market
from .market_activity_service import get_runtime_activity_status

SCAN_BLOCKING_ACTIVITY_STAGES = frozenset({"prices", "fundamentals"})
SCAN_BLOCKING_ACTIVITY_STATUSES = frozenset({"queued", "running"})


class _ClosableSession(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MarketGateAllowed:
    market: Market


@dataclass(frozen=True, slots=True)
class MarketGateConflict:
    market: Market
    detail: dict[str, object]


MarketGateResult = MarketGateAllowed | MarketGateConflict


class MarketActivityGate:
    """Decides whether scan creation is blocked by active market refresh work."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], _ClosableSession] = SessionLocal,
        runtime_activity_reader: Callable[[Any], dict[str, Any]] = get_runtime_activity_status,
        blocking_stages: frozenset[str] = SCAN_BLOCKING_ACTIVITY_STAGES,
        blocking_statuses: frozenset[str] = SCAN_BLOCKING_ACTIVITY_STATUSES,
    ) -> None:
        self._session_factory = session_factory
        self._runtime_activity_reader = runtime_activity_reader
        self._blocking_stages = blocking_stages
        self._blocking_statuses = blocking_statuses

    def check(self, market: Market | str) -> MarketGateResult:
        resolved_market = market if isinstance(market, Market) else Market.from_str(market)
        runtime_activity = self._load_runtime_activity()
        conflicting_activity = [
            item
            for item in runtime_activity.get("markets", [])
            if str(item.get("market", "")).upper() == resolved_market.code
            and item.get("stage_key") in self._blocking_stages
            and item.get("status") in self._blocking_statuses
        ]
        if not conflicting_activity:
            return MarketGateAllowed(resolved_market)

        active_stages = sorted(
            {str(item.get("stage_key")) for item in conflicting_activity if item.get("stage_key")}
        )
        lifecycle = conflicting_activity[0].get("lifecycle")
        stage_labels = ", ".join(
            item.get("stage_label") or str(item.get("stage_key")).replace("_", " ").title()
            for item in conflicting_activity
        )
        return MarketGateConflict(
            market=resolved_market,
            detail={
                "code": "market_refresh_active",
                "message": (
                    f"{resolved_market.code} {stage_labels.lower()} is running or queued. "
                    "Wait for it to finish before starting a scan."
                ),
                "market": resolved_market.code,
                "active_stages": active_stages,
                "lifecycle": lifecycle,
            },
        )

    def _load_runtime_activity(self) -> dict[str, Any]:
        session = self._session_factory()
        try:
            return self._runtime_activity_reader(session)
        finally:
            session.close()

