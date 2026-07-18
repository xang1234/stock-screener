import pytest

from app.services.breadth_coverage import (
    BreadthCalculationResult,
    BreadthCoverageReport,
    BreadthOutcomeCounter,
    BreadthOutcomeReport,
    BreadthPriceCoverageAccumulator,
)


def _price_coverage():
    accumulator = BreadthPriceCoverageAccumulator()
    accumulator.record_batch(["AAA", "BBB", "MISS"], ["MISS"])
    return accumulator.report()


def _build_report(batch_order):
    price_coverage = BreadthPriceCoverageAccumulator()
    for candidates, misses in batch_order:
        price_coverage.record_batch(candidates, misses)
    outcomes = BreadthOutcomeCounter()
    outcomes.record_scanned()
    outcomes.record_insufficient()
    outcomes.record_error()
    return BreadthCoverageReport.from_parts(
        price_coverage.report(),
        outcomes.report(),
    )


def test_composed_report_keeps_unique_symbols_separate_from_observations():
    outcomes = BreadthOutcomeCounter()
    outcomes.record_scanned()
    outcomes.record_insufficient()
    outcomes.record_cache_miss()

    report = BreadthCoverageReport.from_parts(
        _price_coverage(),
        outcomes.report(),
    )

    assert report.candidate_stocks == 3
    assert report.cache_miss_stocks == 1
    assert report.total_stocks_scanned == 1
    assert report.skipped_stocks == 2
    assert report.insufficient_history_observations == 1


def test_many_date_outcomes_share_one_price_coverage_value():
    price_coverage = _price_coverage()
    first = BreadthCoverageReport.from_parts(
        price_coverage,
        BreadthOutcomeReport(scanned=1),
    )
    second = BreadthCoverageReport.from_parts(
        price_coverage,
        BreadthOutcomeReport(cache_misses=1),
    )

    assert first.price_coverage is price_coverage
    assert second.price_coverage is price_coverage


def test_report_derives_counts_from_symbol_identity():
    report = _build_report([
        (["AAA", "MISS2"], ["MISS2"]),
        (["BBB", "MISS1"], ["MISS1"]),
    ])

    assert report.candidate_stocks == 4
    assert report.symbols_with_cached_history == 2
    assert report.cache_miss_stocks == 2
    assert report.cache_coverage_ratio == 0.5
    assert report.total_stocks_scanned == 1
    assert report.skipped_stocks == 2
    assert report.insufficient_data_stocks == 1
    assert report.error_stocks == 1
    assert report.insufficient_history_observations == 1


def test_cache_miss_sample_is_deterministic_across_batch_order():
    forward = _build_report([
        (["ZZZ", "AAA"], ["ZZZ"]),
        (["MMM", "BBB"], ["MMM"]),
    ])
    reverse = _build_report([
        (["MMM", "BBB"], ["MMM"]),
        (["ZZZ", "AAA"], ["ZZZ"]),
    ])

    assert forward.cache_miss_symbols_sample == ("MMM", "ZZZ")
    assert reverse.cache_miss_symbols_sample == forward.cache_miss_symbols_sample


def test_price_coverage_rejects_misses_outside_candidate_batch():
    accumulator = BreadthPriceCoverageAccumulator()

    with pytest.raises(ValueError, match="outside candidate batch"):
        accumulator.record_batch(["AAA"], ["BBB"])

    assert accumulator.report().candidate_stocks == 0


def test_price_coverage_rejects_conflicting_repeated_classification():
    accumulator = BreadthPriceCoverageAccumulator()
    accumulator.record_batch(["AAA", "BBB"], ["BBB"])

    with pytest.raises(ValueError, match="conflicting classification"):
        accumulator.record_batch(["BBB"], [])

    report = accumulator.report()
    assert report.candidate_stocks == 2
    assert report.symbols_with_cached_history == 1
    assert report.cache_miss_stocks == 1
    assert report.cache_coverage_ratio == 0.5


def test_daily_and_backfill_serializers_share_one_report():
    report = _build_report([
        (["AAA", "BBB"], ["BBB"]),
    ])

    assert report.to_daily_dict() == {
        "candidate_stocks": 2,
        "total_stocks_scanned": 1,
        "symbols_with_cached_history": 1,
        "skipped_stocks": 2,
        "cache_miss_stocks": 1,
        "insufficient_data_stocks": 1,
        "error_stocks": 1,
        "cache_coverage_ratio": 0.5,
        "cache_miss_symbols_sample": ["BBB"],
    }
    assert report.to_backfill_dict() == {
        "target_symbols": 2,
        "symbols_with_cached_history": 1,
        "cache_miss_stocks": 1,
        "cache_miss_symbols_sample": ["BBB"],
        "cache_coverage_ratio": 0.5,
        "insufficient_history_observations": 1,
    }


def test_calculation_result_serializes_indicators_and_coverage():
    price_coverage = BreadthPriceCoverageAccumulator()
    price_coverage.record_batch(["AAA"], [])
    outcomes = BreadthOutcomeCounter()
    outcomes.record_scanned()
    result = BreadthCalculationResult(
        indicators={"stocks_up_4pct": 1, "ratio_5day": None},
        coverage=BreadthCoverageReport.from_parts(
            price_coverage.report(),
            outcomes.report(),
        ),
    )

    assert result.to_metrics_dict() == {
        "stocks_up_4pct": 1,
        "ratio_5day": None,
        "candidate_stocks": 1,
        "total_stocks_scanned": 1,
        "symbols_with_cached_history": 1,
        "skipped_stocks": 0,
        "cache_miss_stocks": 0,
        "insufficient_data_stocks": 0,
        "error_stocks": 0,
        "cache_coverage_ratio": 1.0,
        "cache_miss_symbols_sample": [],
    }
