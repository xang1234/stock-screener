"""OpenAPI contract checks for theme analyst workflow endpoints."""

from __future__ import annotations

from app.main import app


def _find_path(paths: dict, suffix: str) -> tuple[str, dict]:
    for path, payload in paths.items():
        if path.endswith(suffix):
            return path, payload
    raise AssertionError(f"OpenAPI path not found for suffix: {suffix}")


def test_theme_openapi_contract_includes_merge_rankings_candidates_and_relationships():
    schema = app.openapi()
    paths = schema.get("paths", {})
    components = schema.get("components", {}).get("schemas", {})

    rankings_path, rankings_ops = _find_path(paths, "/themes/rankings")
    merge_path, merge_ops = _find_path(paths, "/themes/merge-suggestions")
    candidate_queue_path, candidate_queue_ops = _find_path(paths, "/themes/candidates/queue")
    candidate_review_path, candidate_review_ops = _find_path(paths, "/themes/candidates/review")
    merge_plan_path, merge_plan_ops = _find_path(paths, "/themes/merge-plan/dry-run")
    refresh_campaign_path, refresh_campaign_ops = _find_path(paths, "/themes/embeddings/refresh-campaign")
    strict_wave_path, strict_wave_ops = _find_path(paths, "/themes/merge-wave/strict-auto")
    manual_wave_path, manual_wave_ops = _find_path(paths, "/themes/merge-wave/manual-review")
    relationship_graph_path, relationship_graph_ops = _find_path(paths, "/themes/relationship-graph")

    assert rankings_path.endswith("/themes/rankings")
    assert merge_path.endswith("/themes/merge-suggestions")
    assert candidate_queue_path.endswith("/themes/candidates/queue")
    assert candidate_review_path.endswith("/themes/candidates/review")
    assert merge_plan_path.endswith("/themes/merge-plan/dry-run")
    assert refresh_campaign_path.endswith("/themes/embeddings/refresh-campaign")
    assert strict_wave_path.endswith("/themes/merge-wave/strict-auto")
    assert manual_wave_path.endswith("/themes/merge-wave/manual-review")
    assert relationship_graph_path.endswith("/themes/relationship-graph")

    assert rankings_ops["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ThemeRankingsResponse"
    )
    assert merge_ops["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ThemeMergeSuggestionsResponse"
    )
    assert candidate_queue_ops["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/CandidateThemeQueueResponse"
    )
    assert candidate_review_ops["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/CandidateThemeReviewRequest"
    )
    assert candidate_review_ops["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/CandidateThemeReviewResponse"
    )
    assert merge_plan_ops["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/MergePlanDryRunResponse"
    )
    assert refresh_campaign_ops["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/EmbeddingRefreshCampaignResponse"
    )
    assert strict_wave_ops["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/StrictAutoMergeWaveResponse"
    )
    assert manual_wave_ops["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ManualReviewWaveRequest"
    )
    assert manual_wave_ops["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ManualReviewWaveResponse"
    )
    assert relationship_graph_ops["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ThemeRelationshipGraphResponse"
    )

    for required_schema in [
        "ThemeRankingsResponse",
        "ThemeMergeSuggestionsResponse",
        "CandidateThemeQueueResponse",
        "CandidateThemeReviewRequest",
        "CandidateThemeReviewResponse",
        "MergePlanDryRunResponse",
        "EmbeddingRefreshCampaignResponse",
        "StrictAutoMergeWaveResponse",
        "ManualReviewWaveRequest",
        "ManualReviewWaveResponse",
        "ThemeRelationshipGraphResponse",
    ]:
        assert required_schema in components
