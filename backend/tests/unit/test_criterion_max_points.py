"""The built-in scanners and ExplainStockUseCase share one max-points table."""

import pytest

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.domain.scanning.models import ScanResultItemDomain, ScreenerOutputDomain
from app.domain.scanning.ports import CanonicalStockRsSource
from app.domain.scanning.scoring import CRITERION_MAX_POINTS_BY_SCREENER
from app.scanners import screener_registry
from app.use_cases.scanning.explain_stock import ExplainStockUseCase

from tests.unit.conftest import _make_stub_stock_data


@pytest.mark.parametrize(
    "name",
    [
        "canslim",
        "ipo",
        pytest.param(
            "minervini",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "pre-existing: Minervini criteria top out at 90 (70 without "
                    "VCP), yet the score is left unscaled with VCP and divided by "
                    "80 without it, so a perfect stock scores 90 / 87.5"
                ),
            ),
        ),
    ],
)
def test_criterion_maxima_fill_the_100_point_scale(name):
    assert sum(CRITERION_MAX_POINTS_BY_SCREENER[name].values()) == 100


def _stock_data():
    data = _make_stub_stock_data("TEST")
    data.rs_source = CanonicalStockRsSource(
        formula_version=BALANCED_RS_FORMULA_VERSION,
        run_id=1,
        universe_size=5000,
        ratings={"rs_rating": 85, "rs_rating_1m": 80, "rs_rating_3m": 85, "rs_rating_12m": 90},
    )
    return data


@pytest.mark.parametrize("name", sorted(CRITERION_MAX_POINTS_BY_SCREENER))
def test_explain_reports_each_scanned_criterion_against_its_maximum(name):
    result = screener_registry.get(name).scan_stock("TEST", _stock_data())
    item = ScanResultItemDomain(
        symbol="TEST",
        composite_score=result.score,
        rating=result.rating,
        current_price=None,
        screener_outputs={
            name: ScreenerOutputDomain(
                screener_name=name,
                score=result.score,
                passes=result.passes,
                rating=result.rating,
                breakdown=result.breakdown,
                details=result.details,
            )
        },
        screeners_run=[name],
        composite_method="weighted_average",
        screeners_passed=int(result.passes),
        screeners_total=1,
    )

    explanation = ExplainStockUseCase.build_explanation_from_item(item)

    criteria = {c.name: c for c in explanation.screener_explanations[0].criteria}
    for criterion, max_points in CRITERION_MAX_POINTS_BY_SCREENER[name].items():
        assert criteria[criterion].max_score == max_points
        assert 0 <= criteria[criterion].score <= max_points
