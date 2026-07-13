"""Shared freshness/coherence helpers for daily snapshot payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, Mapping, Sequence

KeyMarketMismatchStatus = Literal["missing", "stale", "future"]


def iso_or_none(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    return date.fromisoformat(text) if text else None


@dataclass(frozen=True)
class KeyMarketDateRange:
    min: date
    max: date

    def to_payload(self) -> dict[str, str]:
        return {
            "min": self.min.isoformat(),
            "max": self.max.isoformat(),
        }


@dataclass(frozen=True)
class KeyMarketDateMismatch:
    symbol: str
    latest_date: date | None
    status: KeyMarketMismatchStatus

    def to_payload(self) -> dict[str, str | None]:
        return {
            "symbol": self.symbol,
            "latest_date": iso_or_none(self.latest_date),
            "status": self.status,
        }


@dataclass(frozen=True)
class KeyMarketDateSummary:
    latest_date: date | None
    date_range: KeyMarketDateRange | None
    mismatched_symbols: tuple[KeyMarketDateMismatch, ...]


@dataclass(frozen=True)
class SnapshotSectionDates:
    breadth: date | None = None
    groups: date | None = None
    exposure: date | None = None
    key_markets: date | None = None

    @classmethod
    def from_raw(
        cls,
        *,
        breadth: Any = None,
        groups: Any = None,
        exposure: Any = None,
        key_markets: Any = None,
    ) -> "SnapshotSectionDates":
        return cls(
            breadth=_coerce_date(breadth),
            groups=_coerce_date(groups),
            exposure=_coerce_date(exposure),
            key_markets=_coerce_date(key_markets),
        )

    def values(self) -> tuple[date | None, ...]:
        return (self.breadth, self.groups, self.exposure, self.key_markets)


def key_market_date_summary(
    entries: Sequence[Mapping[str, Any]],
    *,
    anchor: date | None,
) -> KeyMarketDateSummary:
    dates: list[date] = []
    mismatches: list[KeyMarketDateMismatch] = []
    for entry in entries:
        symbol = str(entry.get("symbol") or "").strip()
        latest_date = _coerce_date(entry.get("latest_date"))
        if latest_date is not None:
            dates.append(latest_date)
        if anchor is None:
            continue
        status = None
        if latest_date is None:
            status = "missing"
        elif latest_date > anchor:
            status = "future"
        elif latest_date < anchor:
            status = "stale"
        if status is not None:
            mismatches.append(
                KeyMarketDateMismatch(
                    symbol=symbol,
                    latest_date=latest_date,
                    status=status,
                )
            )

    date_range = None
    if dates:
        date_range = KeyMarketDateRange(min=min(dates), max=max(dates))
    return KeyMarketDateSummary(
        latest_date=max(dates) if dates else None,
        date_range=date_range,
        mismatched_symbols=tuple(mismatches),
    )


def coherence_status(
    *,
    anchor: date | None,
    section_dates: SnapshotSectionDates,
    key_market_summary: KeyMarketDateSummary | None = None,
    groups_applicable: bool = True,
) -> str:
    if anchor is None:
        return "unanchored"
    mismatches = tuple(key_market_summary.mismatched_symbols) if key_market_summary else ()
    if any(match.status == "future" for match in mismatches):
        return "future_section_data"
    applicable_dates = (
        section_dates.values()
        if groups_applicable
        else (section_dates.breadth, section_dates.exposure, section_dates.key_markets)
    )
    if any(value is not None and value > anchor for value in applicable_dates):
        return "future_section_data"
    if all(value == anchor for value in applicable_dates) and not mismatches:
        return "coherent"
    return "partial"


def build_snapshot_freshness(
    *,
    base_freshness: Mapping[str, Any],
    anchor: date | None,
    market_timezone: str | None,
    section_dates: SnapshotSectionDates,
    key_markets: list[dict[str, Any]],
    groups_applicable: bool = True,
) -> dict[str, Any]:
    key_summary = key_market_date_summary(key_markets, anchor=anchor)
    all_section_dates = SnapshotSectionDates(
        breadth=section_dates.breadth,
        groups=section_dates.groups,
        exposure=section_dates.exposure,
        key_markets=key_summary.latest_date,
    )
    return {
        **dict(base_freshness),
        "snapshot_as_of_date": iso_or_none(anchor),
        "market_timezone": market_timezone,
        "breadth_latest_date": iso_or_none(all_section_dates.breadth),
        "groups_latest_date": iso_or_none(all_section_dates.groups),
        "exposure_latest_date": iso_or_none(all_section_dates.exposure),
        "key_markets_latest_date": iso_or_none(key_summary.latest_date),
        "key_markets_date_range": (
            key_summary.date_range.to_payload() if key_summary.date_range else None
        ),
        "key_markets_mismatched_symbols": [
            mismatch.to_payload()
            for mismatch in key_summary.mismatched_symbols
        ],
        "date_coherence_status": coherence_status(
            anchor=anchor,
            section_dates=all_section_dates,
            key_market_summary=key_summary,
            groups_applicable=groups_applicable,
        ),
    }
