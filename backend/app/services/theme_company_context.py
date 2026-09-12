"""Read-only, bounded company context for grounded theme extraction.

The reader intentionally trusts only active ``StockUniverse`` identities and
cached ``StockFundamental`` profile fields that have both provider attribution
and a recent provider timestamp.  It never fetches a profile or guesses a
bare word as a ticker.

``profile_status`` is one of ``identity_only``, ``missing``, ``available``,
``stale``, ``not_yet_available``, ``unattributed``, or ``conflicting``.
``available`` can include a subset of the three business fields; individually
stale, future-dated, or unattributed fields are omitted and recorded as
warnings.  A conflicting profile suppresses every business field for that
company.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse
from app.services.cjk_alias_resolver_service import (
    METHOD_SYMBOL_NORMALIZED,
    METHOD_SYMBOL_PASSTHROUGH,
    resolve_alias,
)
from app.services.symbol_format import normalize_symbol
from app.services.theme_ticker_identity import (
    has_explicit_equity_identity,
    is_commodity_futures_reference,
)

MAX_COMPANIES = 20
MAX_DESCRIPTION_CHARS = 1_500
PROFILE_FRESHNESS = timedelta(days=180)
_CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_$])\$([A-Za-z0-9][A-Za-z0-9.\-]{0,19})(?![A-Za-z0-9.\-])")
_PROFILE_PROVIDERS = {"yfinance", "finviz"}


def build_company_context(
    db,
    text: str,
    *,
    as_of: datetime | None = None,
    identity_only: bool = False,
    resolved_symbols: Iterable[object] = (),
) -> dict[str, list[dict[str, Any]] | list[str]]:
    """Return JSON-safe company facts for explicit, active-universe symbols.

    ``text`` contributes only explicit ``$CASHTAG`` references.  A caller may
    pass symbols already resolved from company names through ``resolved_symbols``;
    those symbols are still normalized and checked against the active universe.
    """
    observed_at = _as_utc(as_of) or datetime.now(timezone.utc)
    warnings: list[str] = []
    requested: list[tuple[str, str]] = []
    seen: set[str] = set()

    for raw in _CASHTAG_RE.findall(text or ""):
        canonical = _canonical_symbol(raw)
        if canonical is None:
            warnings.append(f"invalid_explicit_symbol:{raw}")
            continue
        if canonical.isdigit():
            warnings.append(f"ambiguous_numeric_cashtag:{canonical}")
            continue
        if canonical not in seen:
            requested.append((canonical, "explicit"))
            seen.add(canonical)

    for raw in resolved_symbols:
        value = _symbol_from_resolved_value(raw)
        canonical = _canonical_symbol(value)
        label = str(value or "").strip()
        if canonical is None:
            if label:
                warnings.append(f"invalid_resolved_symbol:{label}")
            continue
        if canonical.isdigit():
            warnings.append(f"ambiguous_numeric_resolved_symbol:{canonical}")
            continue
        if canonical not in seen:
            requested.append((canonical, "resolved_name"))
            seen.add(canonical)

    if not requested:
        return {"companies": [], "warnings": _final_warnings(warnings)}

    symbols = [symbol for symbol, _source in requested]
    with db.no_autoflush:
        universe_rows = {
            row.symbol: row
            for row in db.query(StockUniverse)
            .filter(StockUniverse.symbol.in_(symbols))
            .all()
        }

    selected: list[tuple[StockUniverse, str]] = []
    for symbol, resolution_source in requested:
        row = universe_rows.get(symbol)
        if row is None:
            prefix = "unknown_explicit_symbol" if resolution_source == "explicit" else "unknown_resolved_symbol"
            warnings.append(f"{prefix}:{symbol}")
            continue
        if (
            resolution_source == "explicit"
            and is_commodity_futures_reference(symbol, text)
            and not has_explicit_equity_identity(symbol, text, row.name)
        ):
            warnings.append(f"commodity_futures_symbol:{symbol}")
            continue
        if not row.is_active:
            prefix = "inactive_explicit_symbol" if resolution_source == "explicit" else "inactive_resolved_symbol"
            warnings.append(f"{prefix}:{symbol}")
            continue
        selected.append((row, resolution_source))

    if len(selected) > MAX_COMPANIES:
        warnings.append(f"company_limit_reached:{MAX_COMPANIES}")
        selected = selected[:MAX_COMPANIES]

    fundamentals: dict[str, StockFundamental] = {}
    if not identity_only and selected:
        selected_symbols = [row.symbol for row, _source in selected]
        with db.no_autoflush:
            fundamentals = {
                row.symbol: row
                for row in db.query(StockFundamental)
                .filter(StockFundamental.symbol.in_(selected_symbols))
                .all()
            }

    companies: list[dict[str, Any]] = []
    for row, _resolution_source in selected:
        company, profile_warnings = _company_record(
            row,
            fundamentals.get(row.symbol),
            observed_at=observed_at,
            identity_only=identity_only,
        )
        companies.append(company)
        warnings.extend(profile_warnings)
    return {"companies": companies, "warnings": _final_warnings(warnings)}


def _company_record(
    universe: StockUniverse,
    fundamental: StockFundamental | None,
    *,
    observed_at: datetime,
    identity_only: bool,
) -> tuple[dict[str, Any], list[str]]:
    company: dict[str, Any] = {
        "symbol": universe.symbol,
        "name": universe.name,
        "identity_source": _identity_source(universe),
        "sector": None,
        "industry": None,
        "business_description": None,
        "profile_source": None,
        "profile_as_of": None,
        "profile_status": "identity_only" if identity_only else "missing",
    }
    if identity_only or fundamental is None:
        return company, []

    values, status, warnings = _usable_profile_fields(fundamental, observed_at)
    company["profile_status"] = status
    if values:
        company["sector"] = values.get("sector", (None, None, None))[0]
        company["industry"] = values.get("industry", (None, None, None))[0]
        company["business_description"] = values.get("business_description", (None, None, None))[0]
        company["profile_source"] = {
            field: source for field, (_value, source, _timestamp) in sorted(values.items())
        }
        company["profile_as_of"] = {
            field: _iso(timestamp) for field, (_value, _source, timestamp) in sorted(values.items())
        }
    return company, [f"{warning}:{universe.symbol}" for warning in warnings]


def _usable_profile_fields(
    record: StockFundamental, observed_at: datetime
) -> tuple[dict[str, tuple[str, str, datetime]], str, list[str]]:
    provenance = record.field_provenance if isinstance(record.field_provenance, Mapping) else {}
    candidates = (
        ("sector", _clean_text(record.sector), _field_source(provenance, "sector"), False),
        ("industry", _clean_text(record.industry), _field_source(provenance, "industry"), False),
        *_description_candidates(record, provenance),
    )
    present = [candidate for candidate in candidates if candidate[1] is not None]
    if not present:
        return {}, "missing", []

    # A source attached to a description that disagrees with the column's
    # provider identity is a provenance contradiction, not a safe fallback.
    if any(
        source == "__conflicting__"
        for _field, _value, source, _truncated in present
    ):
        return {}, "conflicting", ["conflicting_profile_provenance"]

    usable: dict[str, tuple[str, str, datetime]] = {}
    stale = False
    future = False
    unattributed = False
    description_truncated = False
    for field, value, source, truncated in present:
        if source is None:
            unattributed = True
            continue
        timestamp = _provider_timestamp(record, source)
        if timestamp is None:
            unattributed = True
            continue
        if timestamp > observed_at:
            future = True
            continue
        if observed_at - timestamp > PROFILE_FRESHNESS:
            stale = True
            continue
        # The preferred finviz description is emitted first; do not replace it
        # with the yfinance fallback when both are current.
        if field not in usable:
            usable[field] = (value, source, timestamp)
            description_truncated = description_truncated or (
                field == "business_description" and truncated
            )

    warnings: list[str] = []
    if future:
        warnings.append("future_profile_fields")
    if stale:
        warnings.append("stale_profile_fields")
    if unattributed:
        warnings.append("unattributed_profile_fields")
    if description_truncated:
        warnings.append("description_truncated")
    if usable:
        return usable, "available", warnings
    if future:
        return {}, "not_yet_available", warnings
    if stale:
        return {}, "stale", warnings
    return {}, "unattributed", warnings


def _description_candidates(
    record: StockFundamental, provenance: Mapping[str, object]
) -> tuple[tuple[str, str | None, str | None, bool], ...]:
    # Finviz is the established cached-description preference.  Both column
    # names carry an inherent provider; optional field provenance may only
    # confirm that provider, never override it.
    return (
        (
            "business_description",
            _bounded_description(record.description_finviz),
            _description_source(provenance, "description_finviz", "finviz"),
            _description_was_truncated(record.description_finviz),
        ),
        (
            "business_description",
            _bounded_description(record.description_yfinance),
            _description_source(provenance, "description_yfinance", "yfinance"),
            _description_was_truncated(record.description_yfinance),
        ),
    )


def _description_source(
    provenance: Mapping[str, object], field: str, expected: str
) -> str | None:
    if field not in provenance:
        return expected
    supplied = _field_source(provenance, field)
    return expected if supplied == expected else "__conflicting__"


def _field_source(provenance: Mapping[str, object], field: str) -> str | None:
    value = provenance.get(field)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _PROFILE_PROVIDERS else None


def _provider_timestamp(record: StockFundamental, source: str) -> datetime | None:
    timestamp = (
        record.yahoo_profile_refreshed_at
        if source == "yfinance"
        else record.finviz_snapshot_at
    )
    if timestamp is None and str(record.data_source or "").strip().lower() == source:
        timestamp = record.data_source_timestamp
    return _as_utc(timestamp)


def _identity_source(record: StockUniverse) -> str:
    source = _clean_text(record.source) or "unknown"
    timestamp = (
        _as_utc(record.last_seen_in_source_at)
        or _as_utc(record.updated_at)
        or _as_utc(record.first_seen_at)
        or _as_utc(record.added_at)
    )
    return f"local_stock_universe:{source}:{_iso(timestamp) if timestamp else 'unknown'}"


def _symbol_from_resolved_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        symbol = value.get("symbol")
        return symbol if isinstance(symbol, str) else None
    symbol = getattr(value, "symbol", None)
    return symbol if isinstance(symbol, str) else None


def _canonical_symbol(value: str | None) -> str | None:
    normalized = normalize_symbol(value)
    if normalized is None:
        return None
    if "." not in normalized:
        return normalized
    resolution = resolve_alias(normalized)
    if resolution.method in {METHOD_SYMBOL_NORMALIZED, METHOD_SYMBOL_PASSTHROUGH}:
        return normalize_symbol(resolution.canonical_symbol)
    return normalized


def _bounded_description(value: object) -> str | None:
    cleaned = _clean_text(value)
    return cleaned[:MAX_DESCRIPTION_CHARS] if cleaned else None


def _description_was_truncated(value: object) -> bool:
    cleaned = _clean_text(value)
    return bool(cleaned and len(cleaned) > MAX_DESCRIPTION_CHARS)


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _final_warnings(values: list[str]) -> list[str]:
    unique = list(dict.fromkeys(values))
    if len(unique) <= 100:
        return unique
    return [*unique[:99], f"warnings_omitted:{len(unique) - 99}"]
