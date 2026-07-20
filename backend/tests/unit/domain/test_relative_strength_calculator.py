import pytest

from app.domain.relative_strength.calculator import (
    BALANCED_RS_FORMULA_VERSION,
    HORIZON_WEIGHTS,
    calculate_balanced_rs,
    percentile_ratings,
)


def test_percentile_ratings_use_average_ties_and_half_up_mapping():
    assert percentile_ratings({"A": 0.0, "B": 10.0, "C": 10.0, "D": 20.0}) == {
        "A": 1,
        "B": 50,
        "C": 50,
        "D": 99,
    }


def test_percentile_ratings_map_an_all_tied_distribution_to_50():
    assert percentile_ratings({"A": 7.0, "B": 7.0, "C": 7.0}) == {
        "A": 50,
        "B": 50,
        "C": 50,
    }


def test_balanced_weights_are_exactly_one():
    assert sum(HORIZON_WEIGHTS.values()) == pytest.approx(1.0)
    assert HORIZON_WEIGHTS == {
        "1m": 0.20,
        "3m": 0.30,
        "6m": 0.20,
        "9m": 0.15,
        "12m": 0.15,
    }


def test_raw_magnitude_cannot_change_a_symbol_after_horizon_rank_is_fixed():
    returns = {
        "A": {"1m": -0.40, "3m": -0.20, "6m": 0.10, "9m": 0.40, "12m": 10.0},
        "B": {"1m": 0.10, "3m": 0.15, "6m": 0.20, "9m": 0.30, "12m": 1.0},
        "C": {"1m": 0.20, "3m": 0.25, "6m": 0.30, "9m": 0.20, "12m": 0.5},
    }
    baseline = calculate_balanced_rs(returns)
    extreme = calculate_balanced_rs(
        {**returns, "A": {**returns["A"], "12m": 10_000.0}}
    )

    assert baseline["A"].rs_12m == 99
    assert extreme["A"].rs_12m == 99
    assert extreme["A"].weighted_composite == baseline["A"].weighted_composite
    assert extreme["A"].overall_rs == baseline["A"].overall_rs


def test_recent_relative_weakness_controls_half_of_the_composite():
    scores = calculate_balanced_rs(
        {
            "FORMER": {"1m": -0.50, "3m": -0.35, "6m": 0.10, "9m": 1.0, "12m": 10.0},
            "STEADY": {"1m": 0.12, "3m": 0.18, "6m": 0.20, "9m": 0.25, "12m": 0.30},
            "MIDDLE": {"1m": 0.02, "3m": 0.04, "6m": 0.08, "9m": 0.12, "12m": 0.15},
        }
    )

    assert scores["FORMER"].rs_1m == 1
    assert scores["FORMER"].rs_3m == 1
    assert scores["FORMER"].weighted_composite < scores["STEADY"].weighted_composite


def test_calculator_requires_one_common_complete_eligible_set():
    with pytest.raises(ValueError, match="missing horizons"):
        calculate_balanced_rs(
            {
                "A": {"1m": 0.1, "3m": 0.2, "6m": 0.3, "9m": 0.4, "12m": 0.5},
                "B": {"1m": 0.1, "3m": 0.2},
            }
        )


def test_formula_version_is_stable():
    assert BALANCED_RS_FORMULA_VERSION == "balanced-horizon-percentile-v2"
