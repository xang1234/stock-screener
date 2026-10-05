"""Serve a scan from one pinned, published per-Market feature run (#492).

A scan of any Universe type except TEST may reuse a published
snapshot only when the answer provably matches what async compute would
produce for the same resolved symbols:

* every symbol resolves to one authoritative Market, and the run is that
  Market's ``latest_published_market:<M>`` publication (read once, then
  referenced by ID, so a pointer move mid-request cannot switch sources);
* every symbol has an actual feature row in that run;
* the run used canonical Market RS (legacy scan-local RS depends on its
  reference Universe, so it is never reused for a subset);
* scoring is equivalent: the same screener profile (scores reused), or
  custom criteria the compiler proves hard-gate equivalent and that read
  only facts stored in the run itself;
* in ``current`` mode, the run is as of the Market's last completed session.

Anything else is ineligible, with a reason code the caller can surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from app.domain.relative_strength.calculator import LEGACY_RS_FORMULA_VERSION
from app.domain.scanning.custom_criteria_compiler import compile_custom_criteria
from app.domain.scanning.signature import (
    build_scan_signature_payload,
    hash_scan_signature,
)
from app.domain.feature_store.models import RunStatus
from app.domain.universe import UniverseType

logger = logging.getLogger(__name__)

DATA_MODE_CURRENT = "current"
DATA_MODE_LAST_PUBLISHED = "last_published"

# ALL qualifies only when its symbols resolve to one Market, where async
# compute also runs in single-market mode.
PINNED_SNAPSHOT_UNIVERSE_TYPES = frozenset({
    UniverseType.ALL.value,
    UniverseType.MARKET.value,
    UniverseType.EXCHANGE.value,
    UniverseType.INDEX.value,
    UniverseType.CUSTOM.value,
})

# Logical fields the feature-store adapters read from the *current*
# StockUniverse / StockFundamental rows rather than from the run.
_UNPINNED_FILTER_FIELDS = frozenset({
    "market", "exchange", "currency", "market_cap_usd", "adv_usd",
})

_REASON_MESSAGES = {
    "unsupported_universe": "This universe type cannot be served from a published snapshot.",
    "unresolved_symbols": "Some symbols have no known market listing.",
    "mixed_market_universe": "The symbols span more than one market; no single publication covers them.",
    "no_published_run": "No published snapshot exists for this market.",
    "source_market_mismatch": "The market's snapshot pointer references another market's run.",
    "snapshot_not_current": "The latest published snapshot is older than the last completed session.",
    "rs_not_canonical": "The published snapshot uses legacy RS, which cannot be reused for a subset.",
    "rs_source_changed": "Market RS has been republished since this snapshot was built.",
    "incomplete_coverage": "The published snapshot is missing rows for some requested symbols.",
    "criteria_not_equivalent": "The scan criteria cannot be answered exactly from the published snapshot.",
    "unpinned_source_facts": "The criteria filter on market cap or dollar volume, which the snapshot does not store.",
    "lookup_failed": "The published snapshot could not be read.",
}


def is_ok_snapshot_row(details: dict) -> bool:
    """Whether a stored row was fully scored (mirrors the snapshot's status).

    Insufficient-history and listing-only rows still carry facts such as
    ``current_price``, but async compute never passes them, so a compiled
    hard gate must not either.
    """
    status = details.get("result_status")
    if status is not None:
        return status == "ok"
    return "error" not in details and details.get("rating") != "Insufficient Data"


@dataclass(frozen=True)
class SnapshotIneligible:
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> str:
        return _REASON_MESSAGES.get(self.reason, self.reason)


@dataclass(frozen=True)
class PinnedSnapshot:
    run: Any  # FeatureRunDomain
    market: str
    match: str  # "profile" (scores reused) | "compiled" (criteria compiled)
    rows: list[tuple[str, dict]]
    passed: int


def published_source_metadata(
    run: Any,
    *,
    match: str,
    data_mode: str,
    market: str | None,
    membership_hash: str,
    membership_count: int,
    expected_session: date | None,
) -> dict[str, Any]:
    """Describe the pinned publication a completed snapshot scan came from.

    ``is_current`` is true when the run was as of ``expected_session``, the
    Market's last completed session when the scan was created; both are
    recorded so later readers can judge the age themselves.
    """
    config = run.config if isinstance(getattr(run, "config", None), dict) else {}
    published_at = getattr(run, "published_at", None)
    return {
        "data_mode": data_mode,
        "match": match,
        "feature_run_id": run.id,
        "market": market,
        "as_of_date": run.as_of_date.isoformat(),
        "published_at": published_at.isoformat() if published_at else None,
        "expected_session": expected_session.isoformat() if expected_session else None,
        "is_current": (run.as_of_date == expected_session) if expected_session else None,
        "membership_hash": membership_hash,
        "membership_count": membership_count,
        "rs_formula_version": config.get("rs_formula_version"),
        "market_rs_run_id": config.get("market_rs_run_id"),
    }


def resolve_pinned_snapshot(
    uow: Any,
    *,
    universe_type: str,
    screeners: list[str],
    composite_method: str,
    criteria: dict | None,
    symbols: list[str],
    session_for: Callable[[str], date | None],
    require_current: bool,
    rs_run_id_for: Callable[[str], int | None] | None = None,
) -> PinnedSnapshot | SnapshotIneligible:
    """Return the pinned snapshot answer, or why there is none.

    ``require_current`` demands that the run is as of ``session_for(market)``
    (the Market's last completed session) and that its Market RS run is
    the one async compute would read now (``rs_run_id_for(market)``); an
    unknown session or RS run makes the request ineligible rather than
    guessing.
    """
    if universe_type not in PINNED_SNAPSHOT_UNIVERSE_TYPES:
        return SnapshotIneligible("unsupported_universe")

    markets = uow.universe.resolve_markets(symbols)
    unresolved = [s for s in symbols if s.upper() not in markets]
    if unresolved:
        return SnapshotIneligible(
            "unresolved_symbols", {"symbols": unresolved[:20], "count": len(unresolved)}
        )
    distinct = sorted(set(markets.values()))
    if len(distinct) != 1:
        return SnapshotIneligible("mixed_market_universe", {"markets": distinct})
    market = distinct[0]

    run = uow.feature_runs.get_latest_published(f"latest_published_market:{market}")
    if run is None or run.status != RunStatus.PUBLISHED:
        return SnapshotIneligible("no_published_run", {"market": market})
    config = run.config if isinstance(run.config, dict) else {}
    source = {"market": market, "feature_run_id": run.id, "as_of_date": run.as_of_date.isoformat()}
    universe = config.get("universe") if isinstance(config.get("universe"), dict) else {}
    run_market = str(config.get("market") or universe.get("market") or "").strip().upper()
    if run_market != market:
        # Never trust the pointer key alone; publish APIs accept any key.
        return SnapshotIneligible("source_market_mismatch", {**source, "run_market": run_market or None})

    if require_current:
        session = session_for(market)
        if session is None or run.as_of_date != session:
            return SnapshotIneligible(
                "snapshot_not_current",
                {**source, "expected_session": session.isoformat() if session else None},
            )

    formula = config.get("rs_formula_version")
    if not formula or formula == LEGACY_RS_FORMULA_VERSION or config.get("market_rs_run_id") is None:
        return SnapshotIneligible("rs_not_canonical", source)

    if require_current and (
        rs_run_id_for is None or rs_run_id_for(market) != config.get("market_rs_run_id")
    ):
        return SnapshotIneligible("rs_source_changed", source)

    if not uow.feature_runs.has_feature_rows_for(run.id, symbols):
        return SnapshotIneligible("incomplete_coverage", source)

    # Same screener profile: the run's stored scores are the answer. The
    # source signature's own universe type is substituted because only the
    # membership differs, and membership is pinned by the symbol filter.
    source_signature = config.get("signature")
    if isinstance(source_signature, dict) and source_signature.get("universe_type"):
        request_hash = hash_scan_signature(build_scan_signature_payload(
            universe_type=source_signature["universe_type"],
            screeners=screeners,
            composite_method=composite_method,
            criteria=criteria,
        ))
        if request_hash == run.input_hash:
            rows = uow.feature_store.query_run_details(run.id, None, symbols=symbols)
            passed = sum(1 for _, details in rows if details.get("passes_template"))
            return PinnedSnapshot(run, market, "profile", rows, passed)

    compiled = compile_custom_criteria(criteria, screeners=screeners, universe_market=market)
    spec = compiled.filter_spec
    if not (
        compiled.is_fully_representable
        and compiled.score_field is not None
        and compiled.hard_gate_equivalent
        and (spec.range_filters or spec.categorical_filters or spec.boolean_filters)
    ):
        return SnapshotIneligible("criteria_not_equivalent", source)
    filter_fields = {
        f.field
        for f in (*spec.range_filters, *spec.categorical_filters, *spec.boolean_filters)
    }
    if filter_fields & _UNPINNED_FILTER_FIELDS:
        return SnapshotIneligible("unpinned_source_facts", source)

    rows = [
        (symbol, details)
        for symbol, details in uow.feature_store.query_run_details(run.id, spec, symbols=symbols)
        if is_ok_snapshot_row(details)
    ]
    return PinnedSnapshot(run, market, "compiled", rows, len(rows))
