"""ExplainStockUseCase — explain why a stock received its composite score.

Provides a per-screener, per-criterion breakdown of an existing scan result.
Reports stored values rather than re-deriving scores.

Business rules:
  1. Verify the scan exists (raise EntityNotFoundError if not)
  2. Normalise the symbol to uppercase (case-insensitive lookup)
  3. Retrieve the scan result item via ScanResultRepository.get_by_symbol()
  4. Build a StockExplanation from the item's screener_outputs
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.uow import UnitOfWork
from app.domain.scanning.models import (
    CriterionResult,
    ScreenerExplanation,
    StockExplanation,
)
from app.domain.scanning.scoring import (
    BUY_THRESHOLD,
    STRONG_BUY_THRESHOLD,
    WATCH_THRESHOLD,
)


# ── Query (input) ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExplainStockQuery:
    """Immutable value object describing the explain-stock request."""

    scan_id: str
    symbol: str

    def __post_init__(self) -> None:
        # Business rule: symbols are case-insensitive.
        object.__setattr__(self, "symbol", self.symbol.upper())


# ── Result (output) ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExplainStockResult:
    """What the use case returns to the caller."""

    explanation: StockExplanation


# ── Use Case ────────────────────────────────────────────────────────────


class ExplainStockUseCase:
    """Explain why a stock received its composite score and rating."""

    # Known max-scores per screener per criterion (from scanner source).
    # Screeners not listed here (e.g. "custom") get max_score=0.0.
    _MAX_SCORES: dict[str, dict[str, float]] = {
        "minervini": {
            "rs_rating": 20,
            "stage": 20,
            "ma_alignment": 15,
            "position_52w": 15,
            "vcp": 20,
        },
        "canslim": {
            "current_earnings": 20,
            "annual_earnings": 15,
            "new_highs": 15,
            "supply_demand": 15,
            "leader": 20,
            "institutional": 15,
        },
        "ipo": {
            "ipo_age": 25,
            "performance_since_ipo": 25,
            "price_stability": 20,
            "volume_patterns": 15,
            "revenue_growth": 15,
        },
        "volume_breakthrough": {
            "five_year_high": 43,
            "one_year_high": 43,
            "since_ipo_high": 43,
        },
    }

    _RATING_THRESHOLDS: dict[str, float] = {
        "Strong Buy": STRONG_BUY_THRESHOLD,
        "Buy": BUY_THRESHOLD,
        "Watch": WATCH_THRESHOLD,
        "Pass": 0.0,
    }

    def execute(
        self, uow: UnitOfWork, query: ExplainStockQuery
    ) -> ExplainStockResult:
        with uow:
            scan = uow.scans.get_by_scan_id(query.scan_id)
            if scan is None:
                raise EntityNotFoundError("Scan", query.scan_id)

            item = uow.scan_results.get_by_symbol(
                scan_id=query.scan_id,
                symbol=query.symbol,
            )
            if item is None:
                raise EntityNotFoundError("ScanResult", query.symbol)

        # Build per-screener explanations
        screener_explanations: list[ScreenerExplanation] = []
        for screener_name, output in item.screener_outputs.items():
            max_scores_lookup = self._MAX_SCORES.get(screener_name, {})

            criteria: list[CriterionResult] = []
            for criterion_name, score_value in output.breakdown.items():
                score = float(score_value)
                criteria.append(
                    CriterionResult(
                        name=criterion_name,
                        score=score,
                        max_score=max_scores_lookup.get(criterion_name, 0.0),
                        passed=score > 0,
                    )
                )

            screener_explanations.append(
                ScreenerExplanation(
                    screener_name=screener_name,
                    score=output.score,
                    passes=output.passes,
                    rating=output.rating,
                    criteria=tuple(criteria),
                )
            )

        explanation = StockExplanation(
            symbol=item.symbol,
            composite_score=item.composite_score,
            rating=item.rating,
            composite_method=item.composite_method,
            screeners_passed=item.screeners_passed,
            screeners_total=item.screeners_total,
            screener_explanations=tuple(screener_explanations),
            rating_thresholds=dict(self._RATING_THRESHOLDS),
        )

        return ExplainStockResult(explanation=explanation)
