"""Pydantic schemas for Theme Discovery API"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# Content Source Schemas
class ContentSourceCreate(BaseModel):
    name: str = Field(..., description="Source name (e.g., 'Doomberg', '@Jesse_Livermore')")
    source_type: str = Field(..., description="Type: substack, twitter, news, reddit")
    url: str = Field(..., description="RSS feed URL, Twitter handle URL, etc.")
    priority: int = Field(50, ge=1, le=100, description="Priority weight (1-100)")
    fetch_interval_minutes: int = Field(60, description="How often to fetch")
    pipelines: list[str] = Field(default=["technical", "fundamental"], description="Pipelines this source feeds into")


class ContentSourceUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Source name")
    source_type: Optional[str] = Field(None, description="Type: substack, twitter, news, reddit")
    url: Optional[str] = Field(None, description="RSS feed URL, Twitter handle URL, etc.")
    priority: Optional[int] = Field(None, ge=1, le=100, description="Priority weight (1-100)")
    fetch_interval_minutes: Optional[int] = Field(None, description="How often to fetch")
    is_active: Optional[bool] = Field(None, description="Whether source is active")
    pipelines: Optional[list[str]] = Field(None, description="Pipelines this source feeds into")


class ContentSourceResponse(BaseModel):
    id: int
    name: str
    source_type: str
    url: str
    priority: int
    fetch_interval_minutes: int
    is_active: bool
    pipelines: Optional[list[str]] = None
    last_fetched_at: Optional[datetime]
    total_items_fetched: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class BrowserCookieInput(BaseModel):
    name: str
    value: str
    domain: str
    path: str = "/"
    secure: bool = False
    httpOnly: bool = False
    sameSite: Optional[str] = None
    expirationDate: Optional[float] = None
    expires: Optional[float] = None


class TwitterSessionStatusResponse(BaseModel):
    authenticated: bool
    status_code: str
    message: str
    profile: str
    storage_state_path: str
    provider: str = "xui"


class TwitterSessionChallengeResponse(BaseModel):
    challenge_id: str
    challenge_token: str
    expires_at: datetime
    ttl_seconds: int


class TwitterSessionImportRequest(BaseModel):
    challenge_id: str = Field(..., min_length=1)
    challenge_token: str = Field(..., min_length=1)
    cookies: list[BrowserCookieInput]
    browser: Optional[str] = None
    extension_version: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


# Theme Cluster Schemas
class ThemeClusterResponse(BaseModel):
    id: int
    name: str = Field(..., min_length=1)
    canonical_key: str = Field(..., min_length=1, pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    display_name: str = Field(..., min_length=1)
    aliases: Optional[list[str]]
    description: Optional[str]
    pipeline: str = Field(..., pattern=r"^(technical|fundamental)$")
    category: Optional[str]
    is_emerging: bool
    is_validated: bool
    lifecycle_state: str = Field(default="candidate", pattern=r"^(candidate|active|dormant|reactivated|retired)$")
    lifecycle_state_updated_at: Optional[datetime] = None
    candidate_since_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    dormant_at: Optional[datetime] = None
    reactivated_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    discovery_source: Optional[str]
    first_seen_at: Optional[datetime]
    last_seen_at: Optional[datetime]
    # L1/L2 taxonomy fields
    parent_cluster_id: Optional[int] = None
    is_l1: bool = False
    taxonomy_level: int = 2

    class Config:
        from_attributes = True


class ThemeConstituentResponse(BaseModel):
    symbol: str
    source: Optional[str]
    confidence: float
    mention_count: int
    correlation_to_theme: Optional[float]
    first_mentioned_at: Optional[datetime]
    last_mentioned_at: Optional[datetime]

    class Config:
        from_attributes = True


class ThemeRelationshipResponse(BaseModel):
    relation_id: int
    relationship_type: str = Field(..., pattern=r"^(subset|related|distinct)$")
    direction: str = Field(..., pattern=r"^(incoming|outgoing)$")
    confidence: float
    provenance: Optional[str] = None
    evidence: Optional[dict] = None
    peer_theme_id: int
    peer_theme_name: Optional[str] = None
    peer_theme_display_name: Optional[str] = None


class ThemeDetailResponse(BaseModel):
    theme: ThemeClusterResponse
    constituents: list[ThemeConstituentResponse]
    metrics: Optional["ThemeMetricsResponse"]
    relationships: list[ThemeRelationshipResponse] = Field(default_factory=list)


# Theme Metrics Schemas
class ThemeMetricsResponse(BaseModel):
    date: datetime
    rank: Optional[int]
    status: Optional[str]
    momentum_score: Optional[float]

    # Mention metrics
    mentions_1d: int
    mentions_7d: int
    mentions_30d: int
    mention_velocity: Optional[float]
    sentiment_score: Optional[float]

    # Price metrics
    basket_return_1d: Optional[float]
    basket_return_1w: Optional[float]
    basket_return_1m: Optional[float]
    basket_rs_vs_spy: Optional[float]

    # Correlation metrics
    avg_internal_correlation: Optional[float]

    # Breadth metrics
    num_constituents: int
    pct_above_50ma: Optional[float]
    pct_above_200ma: Optional[float]

    # Screener metrics
    num_passing_minervini: int
    num_stage_2: int
    avg_rs_rating: Optional[float]

    class Config:
        from_attributes = True


# Theme Rankings Schemas
class ThemeRankingItem(BaseModel):
    theme_cluster_id: int
    rank: int
    theme: str
    status: str
    lifecycle_state: str
    momentum_score: float
    mention_velocity: float
    mentions_7d: int
    basket_rs_vs_spy: float
    basket_return_1w: float
    pct_above_50ma: float
    avg_correlation: float
    num_constituents: int
    top_tickers: list[str]
    first_seen: Optional[str]


class ThemeRankingsResponse(BaseModel):
    date: Optional[str]
    total_themes: int
    pipeline: str = "technical"
    rankings: list[ThemeRankingItem]


# Emerging Themes
class EmergingThemeResponse(BaseModel):
    theme: str
    first_seen: str
    mentions_7d: int
    velocity: float
    sentiment: float
    lifecycle_state: str
    tickers: list[str]


class EmergingThemesResponse(BaseModel):
    count: int
    themes: list[EmergingThemeResponse]


# Alert Schemas
class ThemeAlertResponse(BaseModel):
    id: int
    alert_type: str
    title: str
    description: Optional[str]
    severity: str
    related_tickers: Optional[list[str]]
    metrics: Optional[dict]
    triggered_at: datetime
    is_read: bool

    class Config:
        from_attributes = True


class AlertsResponse(BaseModel):
    total: int
    unread: int
    alerts: list[ThemeAlertResponse]


class ThemeLifecycleTransitionResponse(BaseModel):
    id: int
    theme_cluster_id: int
    theme_name: str
    pipeline: str
    from_state: str
    to_state: str
    actor: str
    job_name: Optional[str]
    rule_version: Optional[str]
    reason: Optional[str]
    transition_metadata: dict = Field(default_factory=dict)
    transitioned_at: Optional[str]
    transition_history_path: str
    runbook_url: str


class ThemeLifecycleTransitionHistoryResponse(BaseModel):
    total: int
    transitions: list[ThemeLifecycleTransitionResponse]


class CandidateThemeQueueItemResponse(BaseModel):
    theme_cluster_id: int
    theme_name: str
    theme_display_name: str
    candidate_since_at: Optional[str]
    avg_confidence_30d: float
    confidence_band: str
    mentions_7d: int
    source_diversity_7d: int
    persistence_days_7d: int
    momentum_score: Optional[float]
    queue_reason: str
    evidence: dict = Field(default_factory=dict)


class CandidateThemeQueueSummaryBandResponse(BaseModel):
    band: str
    count: int


class CandidateThemeQueueResponse(BaseModel):
    total: int
    items: list[CandidateThemeQueueItemResponse]
    confidence_bands: list[CandidateThemeQueueSummaryBandResponse]


class CandidateThemeReviewRequest(BaseModel):
    theme_cluster_ids: list[int]
    action: str = Field(..., pattern=r"^(promote|reject)$")
    actor: Optional[str] = "analyst"
    note: Optional[str] = None


class CandidateThemeReviewItemResult(BaseModel):
    theme_cluster_id: int
    theme_name: Optional[str] = None
    status: str
    reason: Optional[str] = None
    to_state: Optional[str] = None


class CandidateThemeReviewResponse(BaseModel):
    success: bool
    action: str
    updated: int
    skipped: int
    results: list[CandidateThemeReviewItemResult]
    error: Optional[str] = None


# Theme Mentions Schemas (for viewing news sources)
class ThemeMentionDetailResponse(BaseModel):
    mention_id: int
    content_title: Optional[str]
    content_url: Optional[str]
    author: Optional[str]
    published_at: Optional[datetime]
    excerpt: Optional[str]
    sentiment: Optional[str]
    confidence: Optional[float]
    tickers: list[str]
    source_type: str
    source_name: Optional[str]

    class Config:
        from_attributes = True


class ThemeMentionsResponse(BaseModel):
    theme_name: str
    theme_id: int
    total_count: int
    mentions: list[ThemeMentionDetailResponse]


class ThemeMatchMethodDistributionResponse(BaseModel):
    method: str
    count: int
    pct: float


class ThemeMatchDecisionReasonDistributionResponse(BaseModel):
    reason: str
    count: int
    pct: float


class ThemeMatchBandBucketResponse(BaseModel):
    band: str
    count: int
    pct: float
    new_cluster_rate: float
    attach_rate: float


class ThemeMatchTelemetrySliceResponse(BaseModel):
    key: str
    total_mentions: int
    new_cluster_count: int
    attach_count: int
    new_cluster_rate: float
    attach_rate: float
    method_distribution: list[ThemeMatchMethodDistributionResponse]
    decision_reason_distribution: list[ThemeMatchDecisionReasonDistributionResponse]
    confidence_bands: list[ThemeMatchBandBucketResponse]
    score_bands: list[ThemeMatchBandBucketResponse]


class ThemeMatchTelemetryResponse(BaseModel):
    window_days: int
    start_at: Optional[datetime]
    end_at: Optional[datetime]
    pipeline: Optional[str]
    source_type: Optional[str]
    threshold_version: Optional[str]
    total_mentions: int
    new_cluster_count: int
    attach_count: int
    new_cluster_rate: float
    attach_rate: float
    method_distribution: list[ThemeMatchMethodDistributionResponse]
    decision_reason_distribution: list[ThemeMatchDecisionReasonDistributionResponse]
    confidence_bands: list[ThemeMatchBandBucketResponse]
    score_bands: list[ThemeMatchBandBucketResponse]
    by_threshold_version: list[ThemeMatchTelemetrySliceResponse]
    by_source_type: list[ThemeMatchTelemetrySliceResponse]


class ThemePipelineObservabilityMetricsResponse(BaseModel):
    parse_failure_rate: float
    processed_without_mentions_ratio: float
    new_cluster_rate: float
    total_mentions: int
    new_cluster_count: int
    failed_retryable_count: int
    retryable_growth_ratio: float
    retryable_growth_delta: int
    merge_pending_count: int
    merge_reviewed_count: int
    merge_precision_proxy: float
    match_method_mix: dict[str, float]
    merge_status_counts: dict[str, int]


class ThemePipelineObservabilityAlertResponse(BaseModel):
    key: str
    severity: str
    title: str
    description: str
    metric: str
    value: float | int
    threshold: float | int
    runbook_url: str
    likely_causes: list[str]


class ThemePipelineObservabilityResponse(BaseModel):
    generated_at: str
    window_days: int
    pipeline: str
    metrics: ThemePipelineObservabilityMetricsResponse
    alerts: list[ThemePipelineObservabilityAlertResponse]


# Correlation Discovery Schemas
class CorrelationClusterResponse(BaseModel):
    cluster_id: int
    symbols: list[str]
    num_stocks: int
    avg_correlation: float
    industries: dict
    is_cross_industry: bool


class CrossIndustryPairResponse(BaseModel):
    symbol1: str
    industry1: str
    symbol2: str
    industry2: str
    correlation: float


class CorrelationDiscoveryResponse(BaseModel):
    correlation_clusters: list[CorrelationClusterResponse]
    cross_industry_pairs: list[CrossIndustryPairResponse]
    hub_stocks: list[dict]


# Validation Schemas
class ThemeValidationResponse(BaseModel):
    theme: str
    is_valid: bool
    avg_correlation: float
    min_correlation: Optional[float]
    max_correlation: Optional[float]
    num_constituents: int
    outliers: list[dict]


class NewEntrantResponse(BaseModel):
    symbol: str
    correlation: float
    industry: str


# Ingestion Schemas
class IngestionResponse(BaseModel):
    total_sources: int
    total_new_items: int
    sources_fetched: list[dict]
    errors: list[dict]


class ExtractionResponse(BaseModel):
    processed: int
    total_mentions: int
    errors: int
    new_themes: list[str]


# Theme Merge Suggestion Schemas
class ThemeMergeSuggestionResponse(BaseModel):
    id: int
    source_theme_id: int
    source_theme_name: str
    source_aliases: Optional[list[str]]
    target_theme_id: int
    target_theme_name: str
    target_aliases: Optional[list[str]]
    similarity_score: Optional[float]
    llm_confidence: Optional[float]
    relationship_type: Optional[str]
    reasoning: Optional[str]
    suggested_name: Optional[str]
    status: str
    created_at: Optional[str]

    # Legacy compatibility fields (deprecated)
    source_cluster_id: int
    source_name: str
    target_cluster_id: int
    target_name: str
    embedding_similarity: Optional[float]
    llm_reasoning: Optional[str]
    llm_relationship: Optional[str]
    suggested_canonical_name: Optional[str]


class ThemeMergeSuggestionsResponse(BaseModel):
    total: int
    suggestions: list[ThemeMergeSuggestionResponse]


class ThemeMergeHistoryResponse(BaseModel):
    id: int
    source_name: str
    target_name: str
    merge_type: str
    embedding_similarity: Optional[float]
    llm_confidence: Optional[float]
    llm_reasoning: Optional[str]
    constituents_merged: int
    mentions_merged: int
    merged_at: Optional[str]
    merged_by: str


class ThemeMergeHistoryListResponse(BaseModel):
    total: int
    history: list[ThemeMergeHistoryResponse]


class MergePlanPairResponse(BaseModel):
    theme1_id: int
    theme1_name: str
    theme2_id: int
    theme2_name: str
    similarity: float
    llm_confidence: float
    relationship: str
    should_merge: bool
    confidence_tier: str
    risk_bucket: str
    recommendation: str


class MergePlanGroupResponse(BaseModel):
    group_id: str
    confidence_tier: str
    theme_ids: list[int]
    theme_names: list[str]
    pair_count: int
    rationale: str


class MergePlanWaveResponse(BaseModel):
    wave: int
    title: str
    confidence_tier: str
    group_ids: list[str]
    pair_count: int
    recommendation: str


class MergePlanConfidenceBucketResponse(BaseModel):
    tier: str
    count: int


class MergePlanDryRunResponse(BaseModel):
    timestamp: str
    total_pairs_analyzed: int
    confidence_distribution: list[MergePlanConfidenceBucketResponse]
    merge_groups: list[MergePlanGroupResponse]
    waves: list[MergePlanWaveResponse]
    ambiguity_clusters: list[MergePlanPairResponse]
    do_not_merge: list[MergePlanPairResponse]
    manual_review_recommendations: list[str]


class StrictAutoMergeSkipReasonResponse(BaseModel):
    reason: str
    count: int


class StrictAutoMergeActionResponse(BaseModel):
    theme1_id: int
    theme1_name: str
    theme2_id: int
    theme2_name: str
    similarity: float
    llm_confidence: float
    relationship: str
    should_merge: bool
    decision: str
    reason: Optional[str] = None
    target_cluster_id: Optional[int] = None
    source_cluster_id: Optional[int] = None


class StrictAutoMergePolicyResponse(BaseModel):
    allowed_relationship: str
    excluded_relationships: list[str]
    min_similarity: float
    min_llm_confidence: float


class StrictAutoMergeReconciliationResponse(BaseModel):
    active_themes_before: int
    active_themes_after: int
    active_themes_delta: int
    merge_suggestions_before: int
    merge_suggestions_after: int
    merge_suggestions_delta: int
    merge_history_before: int
    merge_history_after: int
    merge_history_delta: int


class WaveReassignmentStatsResponse(BaseModel):
    merges_applied: int
    constituents_reassigned_total: int
    mentions_reassigned_total: int


class WaveRollbackReferenceResponse(BaseModel):
    merge_history_id: int
    source_cluster_id: Optional[int] = None
    target_cluster_id: Optional[int] = None
    source_cluster_name: Optional[str] = None
    target_cluster_name: Optional[str] = None
    merged_at: Optional[str] = None
    merged_by: Optional[str] = None
    rollback_reference: str


class WaveReconciliationPackageResponse(BaseModel):
    package_version: str
    wave_name: str
    pipeline: Optional[str]
    dry_run: bool
    started_at: str
    completed_at: str
    before_counts: dict[str, int]
    after_counts: dict[str, int]
    delta_counts: dict[str, int]
    reassignment_stats: WaveReassignmentStatsResponse
    rollback_references: list[WaveRollbackReferenceResponse]
    artifact_hash: str
    artifact_id: str
    sealed_at: str


class StrictAutoMergeWaveResponse(BaseModel):
    timestamp: str
    pipeline: Optional[str]
    dry_run: bool
    candidate_pairs: int
    eligible_pairs: int
    processed_pairs: int
    auto_merged: int
    errors: list[str]
    skip_reasons: list[StrictAutoMergeSkipReasonResponse]
    merge_actions: list[StrictAutoMergeActionResponse]
    precision_policy: StrictAutoMergePolicyResponse
    reconciliation: StrictAutoMergeReconciliationResponse
    reconciliation_package: WaveReconciliationPackageResponse


class ManualReviewDecisionRequest(BaseModel):
    suggestion_id: int
    action: str = Field(..., pattern=r"^(approve|reject)$")
    reviewer: str = Field(..., min_length=1, max_length=64)
    note: Optional[str] = None


class ManualReviewWaveRequest(BaseModel):
    decisions: list[ManualReviewDecisionRequest]
    sla_target_hours: float = Field(24.0, ge=0.1, le=168.0)
    queue_limit: int = Field(500, ge=1, le=2000)
    dry_run: bool = False


class ManualReviewWaveMetricsResponse(BaseModel):
    throughput_per_hour: float
    agreement_rate: float
    disagreement_rate: float
    sla_breaches: int
    sla_breach_rate: float


class ManualReviewAuditTrailResponse(BaseModel):
    suggestion_id: int
    source_cluster_id: Optional[int] = None
    target_cluster_id: Optional[int] = None
    reviewer: str
    action: str
    note: Optional[str] = None
    llm_recommended_merge: Optional[bool] = None
    reviewed_at: Optional[str] = None
    turnaround_hours: Optional[float] = None
    status: str
    reason: Optional[str] = None
    error: Optional[str] = None


class ManualReviewWaveResponse(BaseModel):
    timestamp: str
    pipeline: Optional[str]
    dry_run: bool
    sla_target_hours: float
    queue_size: int
    reviewed: int
    approved: int
    rejected: int
    skipped: int
    errors: int
    queue_closed: bool
    pending_after: int
    metrics: ManualReviewWaveMetricsResponse
    audit_trail: list[ManualReviewAuditTrailResponse]
    error_messages: list[str]
    reconciliation: StrictAutoMergeReconciliationResponse
    reconciliation_package: WaveReconciliationPackageResponse


class EmbeddingRefreshCampaignMetricsResponse(BaseModel):
    pipeline: Optional[str]
    total_active_themes: int
    themes_with_embedding: int
    themes_needing_refresh: int
    stale_embeddings: int
    version_mismatch_embeddings: int
    coverage_ratio: float
    freshness_ratio: float
    version_consistency_ratio: float


class EmbeddingRefreshCampaignRetryClusterResponse(BaseModel):
    theme_cluster_id: int
    theme: str
    attempts: int
    last_error: Optional[str] = None


class EmbeddingRefreshCampaignErrorBucketResponse(BaseModel):
    error: str
    count: int


class EmbeddingRefreshCampaignRetryTrackingResponse(BaseModel):
    failed_attempts: int
    retry_queue_size: int
    retry_clusters: list[EmbeddingRefreshCampaignRetryClusterResponse]
    error_buckets: list[EmbeddingRefreshCampaignErrorBucketResponse]


class EmbeddingRefreshCampaignGatesResponse(BaseModel):
    min_coverage_ratio: float
    min_freshness_ratio: float
    coverage_met: bool
    freshness_met: bool
    ready_for_merge_waves: bool


class EmbeddingRefreshCampaignPassResponse(BaseModel):
    pass_number: int = Field(..., alias="pass")
    refresh_candidates: int
    refresh_processed: int
    refresh_refreshed: int
    refresh_unchanged: int
    refresh_failed: int
    stale_result: dict
    metrics: EmbeddingRefreshCampaignMetricsResponse

    model_config = ConfigDict(populate_by_name=True)


class EmbeddingRefreshCampaignResponse(BaseModel):
    timestamp: str
    pipeline: Optional[str]
    embedding_model: str
    embedding_model_version: str
    refresh_batch_size: int
    stale_batch_size: int
    stale_max_batches_per_pass: int
    max_passes: int
    passes_executed: int
    initial_metrics: EmbeddingRefreshCampaignMetricsResponse
    passes: list[EmbeddingRefreshCampaignPassResponse]
    final_metrics: EmbeddingRefreshCampaignMetricsResponse
    retry_tracking: EmbeddingRefreshCampaignRetryTrackingResponse
    gates: EmbeddingRefreshCampaignGatesResponse


class SimilarThemeResponse(BaseModel):
    theme_id: int
    name: str
    similarity: float
    aliases: Optional[list[str]]
    category: Optional[str]


class SimilarThemesResponse(BaseModel):
    source_theme_id: int
    source_theme_name: str
    threshold: float
    similar_themes: list[SimilarThemeResponse]


class ThemeRelationshipGraphNodeResponse(BaseModel):
    theme_cluster_id: int
    theme_name: str
    theme_display_name: str
    lifecycle_state: str
    is_root: bool


class ThemeRelationshipGraphEdgeResponse(BaseModel):
    relation_id: int
    source_theme_id: int
    source_theme_name: Optional[str]
    target_theme_id: int
    target_theme_name: Optional[str]
    relationship_type: str = Field(..., pattern=r"^(subset|related|distinct)$")
    confidence: float
    provenance: Optional[str] = None
    evidence: dict = Field(default_factory=dict)


class ThemeRelationshipGraphResponse(BaseModel):
    theme_cluster_id: int
    total_nodes: int
    total_edges: int
    nodes: list[ThemeRelationshipGraphNodeResponse]
    edges: list[ThemeRelationshipGraphEdgeResponse]


class ConsolidationResultResponse(BaseModel):
    timestamp: str
    dry_run: bool
    embeddings_updated: int
    pairs_found: int
    llm_verified: int
    auto_merged: int
    queued_for_review: int
    merge_details: list[dict]
    errors: list[str]


class MergeActionResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    source_name: Optional[str] = None
    target_name: Optional[str] = None
    constituents_merged: Optional[int] = None
    mentions_merged: Optional[int] = None
    idempotency_key: Optional[str] = None
    idempotent_replay: Optional[bool] = None
    warning: Optional[str] = None


# ==================== Content Item Browser Schemas ====================

class ThemeReference(BaseModel):
    """Minimal theme info for content item display"""
    id: int
    name: str


class ContentItemWithThemesResponse(BaseModel):
    """Content item with aggregated theme mention data"""
    id: int
    title: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    source_type: str
    source_name: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    themes: list[ThemeReference] = []
    sentiments: list[str] = []
    primary_sentiment: Optional[str] = None
    tickers: list[str] = []


class ContentItemsListResponse(BaseModel):
    """Paginated response for content items list"""
    total: int
    limit: int
    offset: int
    items: list[ContentItemWithThemesResponse]


# ==================== L1/L2 Taxonomy Schemas ====================

class L1ThemeRankingItem(BaseModel):
    """L1 parent theme with aggregated metrics and child count."""
    id: int
    display_name: str
    canonical_key: str
    category: Optional[str] = None
    description: Optional[str] = None
    lifecycle_state: str = "active"
    activated_at: Optional[str] = None
    num_l2_children: int = 0
    mentions_7d: int = 0
    mentions_30d: int = 0
    num_constituents: int = 0
    momentum_score: Optional[float] = None
    basket_return_1w: Optional[float] = None
    basket_rs_vs_spy: Optional[float] = None
    rank: Optional[int] = None


class L1ThemeRankingsResponse(BaseModel):
    """Paginated L1 theme rankings."""
    total: int
    pipeline: str = "technical"
    rankings: list[L1ThemeRankingItem]


class L1ChildItem(BaseModel):
    """L2 child theme within an L1 group."""
    id: int
    display_name: str
    canonical_key: str
    category: Optional[str] = None
    lifecycle_state: str = "candidate"
    l1_assignment_method: Optional[str] = None
    l1_assignment_confidence: Optional[float] = None
    mentions_7d: int = 0
    mentions_30d: int = 0
    num_constituents: int = 0
    momentum_score: Optional[float] = None


class L1ThemeDetail(BaseModel):
    """L1 theme metadata."""
    id: int
    display_name: str
    canonical_key: str
    category: Optional[str] = None
    description: Optional[str] = None


class L1ChildrenResponse(BaseModel):
    """L1 theme with paginated L2 children."""
    l1: L1ThemeDetail
    children: list[L1ChildItem]
    total_children: int


class L1CategoryItem(BaseModel):
    """Category with L1 theme count."""
    category: str
    count: int


class L1CategoriesResponse(BaseModel):
    """List of L1 categories."""
    categories: list[L1CategoryItem]


class L2ReassignRequest(BaseModel):
    """Request to reassign an L2 theme to a different L1 parent."""
    l1_id: int = Field(..., description="Target L1 theme ID")


class TaxonomyAssignmentRequest(BaseModel):
    """Request to run taxonomy assignment pipeline."""
    dry_run: bool = Field(True, description="Preview assignments without applying")
    pipeline: str = Field("technical", description="Pipeline to assign")


class UnassignedThemeItem(BaseModel):
    """L2 theme without L1 parent."""
    id: int
    display_name: str
    canonical_key: str
    category: Optional[str] = None
    lifecycle_state: str = "candidate"


class UnassignedThemesResponse(BaseModel):
    """Paginated list of L2 themes without L1 assignment."""
    total: int
    themes: list[UnassignedThemeItem]


# Update ThemeDetailResponse to resolve forward reference
ThemeDetailResponse.model_rebuild()
