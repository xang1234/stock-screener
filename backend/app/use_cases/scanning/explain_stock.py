"""ExplainStockUseCase — explain why a stock received its composite score.

Provides a per-screener, per-criterion breakdown of an existing scan result.
Reports stored values rather than re-deriving scores.

Business rules:
  1. Verify the scan exists (raise EntityNotFoundError if not)
  2. Normalise the symbol to uppercase (case-insensitive lookup)
  3. If bound to a feature run → retrieve FeatureRow from feature store
  4. Otherwise → retrieve details blob from scan_results
  5. Build a StockExplanation from the screener_outputs
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.uow import UnitOfWork
from app.domain.feature_store.models import (
    INT_TO_RATING,
    FeatureRow,
    extract_screener_outputs,
)
from app.domain.scanning.models import (
    CriterionResult,
    ScanResultItemDomain,
    ScreenerExplanation,
    StockExplanation,
)
from app.domain.scanning.scoring import (
    BUY_THRESHOLD,
    STRONG_BUY_THRESHOLD,
    WATCH_THRESHOLD,
)

from ._resolve import resolve_scan

logger = logging.getLogger(__name__)


def _normalize_composite_score(value: object) -> float | None:
    """Clamp stored composite scores into range while preserving unscored rows."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return max(0.0, min(100.0, numeric))


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

    @classmethod
    def build_explanation_from_item(
        cls,
        item: ScanResultItemDomain,
    ) -> StockExplanation:
        """Build a domain explanation from a prepared scan-result item."""

        screener_explanations: list[ScreenerExplanation] = []
        for screener_name, output in item.screener_outputs.items():
            max_scores_lookup = cls._MAX_SCORES.get(screener_name, {})

            criteria: list[CriterionResult] = []
            for criterion_name, raw_value in output.breakdown.items():
                if isinstance(raw_value, dict):
                    score = float(raw_value.get("points", 0))
                    max_score = float(
                        raw_value.get(
                            "max_points",
                            max_scores_lookup.get(criterion_name, 0.0),
                        )
                    )
                    passed = raw_value.get("passes", score > 0)
                else:
                    score = float(raw_value)
                    max_score = max_scores_lookup.get(criterion_name, 0.0)
                    passed = score > 0

                criteria.append(
                    CriterionResult(
                        name=criterion_name,
                        score=score,
                        max_score=max_score,
                        passed=passed,
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

        return StockExplanation(
            symbol=item.symbol,
            composite_score=item.composite_score,
            rating=item.rating,
            composite_method=item.composite_method,
            screeners_passed=item.screeners_passed,
            screeners_total=item.screeners_total,
            screener_explanations=tuple(screener_explanations),
            rating_thresholds=dict(cls._RATING_THRESHOLDS),
        )

    @staticmethod
    def _build_item_from_feature_row(row: FeatureRow) -> ScanResultItemDomain:
        """Reconstruct ScanResultItemDomain from FeatureRow with screener_outputs."""
        d = row.details or {}
        screener_outputs = extract_screener_outputs(d)
        return ScanResultItemDomain(
            symbol=row.symbol,
            composite_score=_normalize_composite_score(row.composite_score),
            rating=INT_TO_RATING.get(row.overall_rating, d.get("rating", "Pass")),
            current_price=d.get("current_price"),
            screener_outputs=screener_outputs,
            screeners_run=d.get("screeners_run", []),
            composite_method=d.get("composite_method", "weighted_average"),
            screeners_passed=d.get("screeners_passed", 0),
            screeners_total=d.get("screeners_total", 0),
        )

    @staticmethod
    def _build_item_from_details(
        symbol: str, details: dict,
    ) -> ScanResultItemDomain:
        """Reconstruct ScanResultItemDomain from a raw details blob (legacy path)."""
        screener_outputs = extract_screener_outputs(details)
        return ScanResultItemDomain(
            symbol=symbol,
            composite_score=_normalize_composite_score(details.get("composite_score")),
            rating=details.get("rating", "Pass"),
            current_price=details.get("current_price"),
            screener_outputs=screener_outputs,
            screeners_run=details.get("screeners_run", []),
            composite_method=details.get("composite_method", "weighted_average"),
            screeners_passed=details.get("screeners_passed", 0),
            screeners_total=details.get("screeners_total", 0),
        )

    def execute(
        self, uow: UnitOfWork, query: ExplainStockQuery
    ) -> ExplainStockResult:
        with uow:
            scan, run_id = resolve_scan(uow, query.scan_id)

            if run_id:
                row = uow.feature_store.get_row_by_symbol(
                    run_id, query.symbol
                )
                if row is None:
                    raise EntityNotFoundError("ScanResult", query.symbol)
                item = self._build_item_from_feature_row(row)
            else:
                logger.info(
                    "Scan %s: reading details for %s from scan_results (no feature run)",
                    query.scan_id,
                    query.symbol,
                )
                details = uow.scan_results.get_details_by_symbol(
                    query.scan_id, query.symbol
                )
                if details is None:
                    raise EntityNotFoundError("ScanResult", query.symbol)
                item = self._build_item_from_details(query.symbol, details)

        return ExplainStockResult(
            explanation=self.build_explanation_from_item(item)
        )
