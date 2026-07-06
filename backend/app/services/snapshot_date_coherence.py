"""Shared freshness/coherence helpers for daily snapshot payloads."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping


def iso_or_none(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _date_text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def key_market_date_summary(
    entries: list[dict[str, Any]],
    *,
    anchor: date | None,
) -> dict[str, Any]:
    dates: list[str] = []
    mismatches: list[dict[str, Any]] = []
    expected = iso_or_none(anchor)
    for entry in entries:
        symbol = str(entry.get("symbol") or "").strip()
        latest_date = _date_text(entry.get("latest_date"))
        if latest_date is not None:
            dates.append(latest_date)
        if expected is None:
            continue
        status = None
        if latest_date is None:
            status = "missing"
        elif latest_date > expected:
            status = "future"
        elif latest_date < expected:
            status = "stale"
        if status is not None:
            mismatches.append(
                {
                    "symbol": symbol,
                    "latest_date": latest_date,
                    "status": status,
                }
            )

    date_range = None
    if dates:
        date_range = {"min": min(dates), "max": max(dates)}
    return {
        "latest_date": max(dates) if dates else None,
        "date_range": date_range,
        "mismatched_symbols": mismatches,
    }


def coherence_status(
    *,
    anchor: date | None,
    section_dates: Mapping[str, str | None],
    key_market_summary: Mapping[str, Any] | None = None,
) -> str:
    if anchor is None:
        return "unanchored"
    expected = anchor.isoformat()
    mismatches = list((key_market_summary or {}).get("mismatched_symbols") or [])
    if any(match.get("status") == "future" for match in mismatches):
        return "future_section_data"
    if any(value and value > expected for value in section_dates.values()):
        return "future_section_data"
    if all(value == expected for value in section_dates.values()) and not mismatches:
        return "coherent"
    return "partial"


def build_snapshot_freshness(
    *,
    base_freshness: Mapping[str, Any],
    anchor: date | None,
    market_timezone: str | None,
    section_dates: Mapping[str, str | None],
    key_markets: list[dict[str, Any]],
) -> dict[str, Any]:
    key_summary = key_market_date_summary(key_markets, anchor=anchor)
    all_section_dates = {
        "breadth": section_dates.get("breadth"),
        "groups": section_dates.get("groups"),
        "exposure": section_dates.get("exposure"),
        "key_markets": key_summary["latest_date"],
    }
    return {
        **dict(base_freshness),
        "snapshot_as_of_date": iso_or_none(anchor),
        "market_timezone": market_timezone,
        "breadth_latest_date": all_section_dates["breadth"],
        "groups_latest_date": all_section_dates["groups"],
        "exposure_latest_date": all_section_dates["exposure"],
        "key_markets_latest_date": key_summary["latest_date"],
        "key_markets_date_range": key_summary["date_range"],
        "key_markets_mismatched_symbols": key_summary["mismatched_symbols"],
        "date_coherence_status": coherence_status(
            anchor=anchor,
            section_dates=all_section_dates,
            key_market_summary=key_summary,
        ),
    }
