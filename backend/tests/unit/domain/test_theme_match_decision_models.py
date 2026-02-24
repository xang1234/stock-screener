"""Unit tests for theme matching decision value objects."""

from app.domain.theme_matching import MatchDecision, MatchThresholdConfig


def test_threshold_config_resolves_pipeline_source_override_first():
    config = MatchThresholdConfig(
        version="v-test",
        default_threshold=0.9,
        pipeline_overrides={"technical": 0.8},
        source_type_overrides={"news": 0.7},
        pipeline_source_type_overrides={"technical": {"news": 0.6}},
    )

    assert config.resolve_threshold(pipeline="technical", source_type="news") == 0.6
    assert config.resolve_threshold(pipeline="technical", source_type="substack") == 0.8
    assert config.resolve_threshold(pipeline="fundamental", source_type="news") == 0.7
    assert config.resolve_threshold(pipeline="fundamental", source_type="reddit") == 0.9


def test_match_decision_is_typed_and_immutable():
    decision = MatchDecision(
        selected_cluster_id=10,
        method="exact_alias_key",
        score=1.0,
        threshold=0.95,
        threshold_version="match-v1",
        fallback_reason=None,
        best_alternative_cluster_id=9,
        best_alternative_score=0.84,
        score_margin=0.16,
    )

    assert decision.selected_cluster_id == 10
    assert decision.method == "exact_alias_key"
    assert decision.threshold_version == "match-v1"
