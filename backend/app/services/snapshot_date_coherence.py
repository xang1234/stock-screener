"""Shared date-coherence helpers for daily snapshot payloads."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping


def iso_or_none(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def latest_key_market_date(entries: list[dict[str, Any]]) -> str | None:
    dates = [entry.get("latest_date") for entry in entries if entry.get("latest_date")]
    return max(dates) if dates else None


def coherence_status(
    *,
    anchor: date | None,
    section_dates: Mapping[str, str | None],
) -> str:
    if anchor is None:
        return "unanchored"
    expected = anchor.isoformat()
    if all(value == expected for value in section_dates.values()):
        return "coherent"
    if any(value and value > expected for value in section_dates.values()):
        return "future_section_data"
    return "partial"
