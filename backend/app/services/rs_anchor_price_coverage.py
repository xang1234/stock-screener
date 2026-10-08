"""Exact adjusted-close sessions canonical stock RS needs, for every RS market.

Group-history coverage only checks markets with group rankings, so AU and DE
went unchecked and lost RS for most symbols when a missing session became the
21-session anchor (#539). This covers the anchors of the as-of session and of
the next ``lookahead_sessions`` sessions, so a hole is repaired before it
becomes an anchor. Only gaps count: a symbol whose stored history starts after
an anchor is a short history, not a hole.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.domain.relative_strength import HORIZON_SESSIONS
from app.models.stock import StockPrice
from app.services.price_value_policy import is_usable_adjusted_close

# About two trading weeks: a hole is repaired at least this long before any
# horizon's anchor reaches it.
RS_ANCHOR_LOOKAHEAD_SESSIONS = 10
# Operator repair mode: every session the longest horizon can anchor on.
RS_ANCHOR_FULL_WINDOW_LOOKAHEAD_SESSIONS = max(HORIZON_SESSIONS.values())
_CHUNK = 500


@dataclass(frozen=True)
class RsAnchorGaps:
    missing_dates_by_symbol: Mapping[str, frozenset[date]] = field(default_factory=dict)

    def count_by_date(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for dates in self.missing_dates_by_symbol.values():
            for day in dates:
                counts[day.isoformat()] = counts.get(day.isoformat(), 0) + 1
        return dict(sorted(counts.items()))


class RsAnchorPriceCoverageService:
    def __init__(self, *, calendar_service) -> None:
        self._calendar_service = calendar_service

    def required_dates(
        self,
        *,
        market: str,
        through_date: date,
        lookahead_sessions: int = RS_ANCHOR_LOOKAHEAD_SESSIONS,
    ) -> frozenset[date]:
        # Offset k of the session j ahead is offset k - j today, so the
        # lookahead needs no future calendar.
        offsets = sorted(
            {
                offset - step
                for offset in HORIZON_SESSIONS.values()
                for step in range(lookahead_sessions + 1)
                if offset - step >= 1
            }
        )
        anchors = self._calendar_service.session_anchors(
            market, through_date, offsets=tuple(offsets)
        )
        return frozenset(day for day in anchors.values() if day != through_date)

    def gaps(
        self,
        db: Session,
        *,
        symbols: Sequence[str],
        required_dates: Collection[date],
    ) -> RsAnchorGaps:
        required = frozenset(required_dates)
        symbols = tuple(dict.fromkeys(symbols))
        if not required or not symbols:
            return RsAnchorGaps()
        available: dict[str, set[date]] = {}
        first: dict[str, date] = {}
        for start in range(0, len(symbols), _CHUNK):
            chunk = symbols[start : start + _CHUNK]
            for symbol, day, adj_close in (
                db.query(StockPrice.symbol, StockPrice.date, StockPrice.adj_close)
                .filter(StockPrice.symbol.in_(chunk), StockPrice.date.in_(required))
                .all()
            ):
                if is_usable_adjusted_close(adj_close):
                    available.setdefault(symbol, set()).add(day)
            first.update(
                db.query(StockPrice.symbol, func.min(StockPrice.date))
                .filter(
                    StockPrice.symbol.in_(chunk),
                    StockPrice.adj_close.isnot(None),
                    StockPrice.adj_close > 0,
                )
                .group_by(StockPrice.symbol)
                .all()
            )
        missing: dict[str, frozenset[date]] = {}
        for symbol in symbols:
            if symbol not in first:
                continue
            holes = frozenset(
                day
                for day in required
                if day > first[symbol] and day not in available.get(symbol, ())
            )
            if holes:
                missing[symbol] = holes
        return RsAnchorGaps(missing)
