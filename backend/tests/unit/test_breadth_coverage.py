from app.services.breadth_coverage import (
    BreadthCalculationResult,
    BreadthCoverageAccumulator,
)


def _build_report(batch_order):
    coverage = BreadthCoverageAccumulator()
    for candidates, misses in batch_order:
        coverage.record_price_batch(candidates, misses)
    coverage.record_scanned()
    coverage.record_insufficient()
    coverage.record_error()
    return coverage.report()


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
    coverage = BreadthCoverageAccumulator()
    coverage.record_price_batch(["AAA"], [])
    coverage.record_scanned()
    result = BreadthCalculationResult(
        indicators={"stocks_up_4pct": 1, "ratio_5day": None},
        coverage=coverage.report(),
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
