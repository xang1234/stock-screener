"""Pydantic schemas for Theme Discovery API"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


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
    source_cluster_id: int
    source_name: str
    source_aliases: Optional[list[str]]
    target_cluster_id: int
    target_name: str
    target_aliases: Optional[list[str]]
    embedding_similarity: Optional[float]
    llm_confidence: Optional[float]
    llm_reasoning: Optional[str]
    llm_relationship: Optional[str]
    suggested_canonical_name: Optional[str]
    status: str
    created_at: Optional[str]


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


# Update ThemeDetailResponse to resolve forward reference
ThemeDetailResponse.model_rebuild()
