"""Read pinned feature/fundamental inputs for the options cohort."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.domain.options_analytics.models import OptionCandidate, OptionCandidateInput
from app.domain.options_analytics.selection import select_current_candidates
from app.domain.scanning.leadership_policy import (
    LEADERS_MAX_GROUP_RANK,
    LEADERS_MIN_RS_RATING,
)
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.stock import StockFundamental, StockPrice


@dataclass(frozen=True)
class CandidateSourceSnapshot:
    source_feature_run_id: int
    as_of_date: date
    top_candidate_inputs: tuple[OptionCandidateInput, ...]
    leader_inputs: tuple[OptionCandidateInput, ...]
    current_candidates: tuple[OptionCandidate, ...]


def _details(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class SqlOptionsCandidateSource:
    def __init__(self, session: Session) -> None:
        self._session = session

    def read(self, source_feature_run_id: int) -> CandidateSourceSnapshot:
        run = self._session.get(FeatureRun, source_feature_run_id)
        if run is None or run.status != "published":
            raise LookupError(
                f"Published feature run {source_feature_run_id} does not exist"
            )
        rows = (
            self._session.query(StockFeatureDaily)
            .filter(StockFeatureDaily.run_id == source_feature_run_id)
            .all()
        )
        symbols = [feature.symbol for feature in rows]
        closes = self._price_closes(symbols, run.as_of_date)
        candidates: list[OptionCandidateInput] = []
        leaders: list[OptionCandidateInput] = []
        for feature in rows:
            details = _details(feature.details_json)
            dollar_volume = _number(details.get("avg_dollar_volume"))
            item = OptionCandidateInput(
                symbol=feature.symbol,
                composite_score=_number(feature.composite_score),
                daily_dollar_volume=_number(dollar_volume),
                spot_price=_number(details.get("current_price")),
                dividend_yield=_number(details.get("dividend_yield")),
                price_closes=closes.get(feature.symbol.strip().upper(), ()),
            )
            candidates.append(item)
            rs_rating = _number(details.get("rs_rating"))
            group_rank = _number(details.get("ibd_group_rank"))
            if (
                rs_rating is not None
                and rs_rating >= LEADERS_MIN_RS_RATING
                and group_rank is not None
                and group_rank <= LEADERS_MAX_GROUP_RANK
            ):
                leaders.append(item)
        current = select_current_candidates(candidates, leaders)
        return CandidateSourceSnapshot(
            source_feature_run_id=run.id,
            as_of_date=run.as_of_date,
            top_candidate_inputs=tuple(candidates),
            leader_inputs=tuple(leaders),
            current_candidates=tuple(current),
        )

    def _price_closes(
        self, symbols: list[str], as_of_date: date
    ) -> dict[str, tuple[float, ...]]:
        canonical_symbols = sorted({symbol.strip().upper() for symbol in symbols})
        if not canonical_symbols:
            return {}
        rows = (
            self._session.query(StockPrice.symbol, StockPrice.close)
            .filter(
                func.upper(StockPrice.symbol).in_(canonical_symbols),
                StockPrice.date <= as_of_date,
                StockPrice.close.isnot(None),
            )
            .order_by(StockPrice.symbol, StockPrice.date.desc())
            .all()
        )
        newest_first: dict[str, list[float]] = {}
        for symbol, close in rows:
            values = newest_first.setdefault(symbol.strip().upper(), [])
            if len(values) < 21:
                values.append(float(close))
        return {
            symbol: tuple(reversed(values))
            for symbol, values in newest_first.items()
        }

    def read_continuity_inputs(
        self, symbols: list[str] | tuple[str, ...], as_of_date: date
    ) -> dict[str, OptionCandidateInput]:
        canonical_symbols = sorted({symbol.strip().upper() for symbol in symbols})
        if not canonical_symbols:
            return {}
        fundamentals = {
            row.symbol.strip().upper(): row
            for row in self._session.query(StockFundamental)
            .filter(func.upper(StockFundamental.symbol).in_(canonical_symbols))
            .all()
        }
        closes = self._price_closes(canonical_symbols, as_of_date)
        result: dict[str, OptionCandidateInput] = {}
        for symbol in canonical_symbols:
            price_closes = closes.get(symbol, ())
            if not price_closes:
                continue
            fundamental = fundamentals.get(symbol)
            result[symbol] = OptionCandidateInput(
                symbol=symbol,
                composite_score=None,
                daily_dollar_volume=(
                    _number(fundamental.adv_usd) if fundamental is not None else None
                ),
                spot_price=price_closes[-1],
                dividend_yield=(
                    _number(fundamental.dividend_yield)
                    if fundamental is not None
                    else None
                ),
                price_closes=price_closes,
            )
        return result
