"""Theme discovery models for tracking market themes from unstructured sources"""
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    Text,
    Boolean,
    Index,
    UniqueConstraint,
    CheckConstraint,
    ForeignKey,
    JSON,
)
from sqlalchemy.sql import func
from ..database import Base


class ContentSource(Base):
    """Configured content sources for theme extraction (Substack, Twitter, News)"""

    __tablename__ = "content_sources"

    id = Column(Integer, primary_key=True, index=True)

    # Source identification
    name = Column(String(100), nullable=False)  # e.g., "Pelham Smithers", "@Jesse_Livermore"
    source_type = Column(String(20), nullable=False, index=True)  # substack, twitter, news, reddit
    url = Column(String(500))  # RSS feed URL, Twitter handle, etc.

    # Configuration
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=50)  # 1-100, higher = more weight
    fetch_interval_minutes = Column(Integer, default=60)  # How often to check

    # Pipeline assignment - which pipelines this source feeds into
    # JSON array: ["technical", "fundamental"] or ["technical"] or ["fundamental"]
    pipelines = Column(JSON, default=["technical", "fundamental"])

    # Stats
    last_fetched_at = Column(DateTime(timezone=True))
    total_items_fetched = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("source_type", "url", name="uix_source_type_url"),
    )


class ContentItem(Base):
    """Raw content items fetched from sources (before LLM extraction)"""

    __tablename__ = "content_items"

    id = Column(Integer, primary_key=True, index=True)

    # Source reference
    source_id = Column(Integer, index=True)  # FK to content_sources
    source_type = Column(String(20), nullable=False, index=True)
    source_name = Column(String(100))

    # Content
    external_id = Column(String(200))  # Original ID from source (for dedup)
    title = Column(String(500))
    content = Column(Text)  # Full text content
    url = Column(String(500))
    author = Column(String(100))

    # Timestamps
    published_at = Column(DateTime(timezone=True), index=True)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())

    # Processing status
    is_processed = Column(Boolean, default=False, index=True)
    processed_at = Column(DateTime(timezone=True))
    extraction_error = Column(Text)  # If extraction failed

    __table_args__ = (
        UniqueConstraint("source_type", "external_id", name="uix_source_external_id"),
        Index("idx_content_unprocessed", "is_processed", "published_at"),
    )


class ThemeMention(Base):
    """Individual theme mentions extracted from content via LLM"""

    __tablename__ = "theme_mentions"

    id = Column(Integer, primary_key=True, index=True)

    # Source reference
    content_item_id = Column(Integer, index=True)  # FK to content_items
    source_type = Column(String(20), nullable=False, index=True)
    source_name = Column(String(100))

    # Extracted theme
    raw_theme = Column(String(200), nullable=False)  # Original theme text from LLM
    canonical_theme = Column(String(200), index=True)  # Normalized theme name
    theme_cluster_id = Column(Integer, index=True)  # FK to theme_clusters (assigned after clustering)
    match_method = Column(String(40), index=True)  # exact_canonical_key, exact_alias_key, exact_display_name, create_new_cluster
    match_score = Column(Float)  # Decision score for selected cluster
    match_threshold = Column(Float)  # Effective threshold used for the decision
    threshold_version = Column(String(40), index=True)  # Threshold config version used
    match_score_model = Column(String(80), index=True)  # Score/model namespace (e.g. all-MiniLM-L6-v2)
    match_score_model_version = Column(String(40), index=True)  # Policy/model version used for comparability
    match_fallback_reason = Column(String(120))  # Why fallback path was used
    best_alternative_cluster_id = Column(Integer, index=True)  # Best rejected candidate cluster (if any)
    best_alternative_score = Column(Float)  # Score for best rejected candidate
    match_score_margin = Column(Float)  # selected_score - best_alternative_score

    # Pipeline assignment - which pipeline extracted this mention
    pipeline = Column(String(20), index=True)  # technical or fundamental

    # Extracted tickers
    tickers = Column(JSON)  # List of tickers: ["NVDA", "AVGO", "MRVL"]
    ticker_count = Column(Integer, default=0)

    # Sentiment & confidence
    sentiment = Column(String(20))  # bullish, bearish, neutral
    confidence = Column(Float)  # 0-1, LLM's confidence in extraction

    # Context
    excerpt = Column(Text)  # Relevant excerpt from source

    # Timestamps
    mentioned_at = Column(DateTime(timezone=True), index=True)  # When content was published
    extracted_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_theme_mention_date", "canonical_theme", "mentioned_at"),
        Index("idx_mention_cluster", "theme_cluster_id", "mentioned_at"),
        Index("idx_theme_mentions_pipeline", "pipeline"),
    )


class ContentItemPipelineState(Base):
    """Per-pipeline processing state for each content item."""

    __tablename__ = "content_item_pipeline_state"

    id = Column(Integer, primary_key=True, index=True)
    content_item_id = Column(Integer, ForeignKey("content_items.id", ondelete="CASCADE"), nullable=False)
    pipeline = Column(String(20), nullable=False)
    status = Column(String(30), nullable=False, default="pending")
    attempt_count = Column(Integer, nullable=False, default=0)
    error_code = Column(String(100))
    error_message = Column(Text)
    last_attempt_at = Column(DateTime(timezone=True))
    processed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("content_item_id", "pipeline", name="uix_cips_content_item_pipeline"),
        CheckConstraint(
            "pipeline IN ('technical', 'fundamental')",
            name="ck_cips_pipeline_values",
        ),
        CheckConstraint(
            "status IN ('pending', 'in_progress', 'processed', 'failed_retryable', 'failed_terminal')",
            name="ck_cips_status_values",
        ),
        Index("idx_cips_pipeline_status_last_attempt", "pipeline", "status", "last_attempt_at"),
        Index("idx_cips_pipeline_status_created", "pipeline", "status", "created_at"),
        Index("idx_cips_content_item_pipeline_status", "content_item_id", "pipeline", "status"),
        Index("idx_cips_updated_at", "updated_at"),
    )


class ThemeCluster(Base):
    """Discovered theme clusters (groups of similar themes)"""

    __tablename__ = "theme_clusters"

    id = Column(Integer, primary_key=True, index=True)

    # Cluster identity
    name = Column(String(200), nullable=False, index=True)  # Legacy name field; mirrors display_name during cutover
    canonical_key = Column(String(96), nullable=False, index=True)  # Pipeline-scoped canonical identity key
    display_name = Column(String(200), nullable=False)  # Human-friendly label independent from canonical_key
    aliases = Column(JSON)  # List of alternative names that map to this cluster
    description = Column(Text)  # Auto-generated or manual description

    # Pipeline assignment - which pipeline this theme belongs to
    pipeline = Column(String(20), nullable=False, default="technical", index=True)  # technical or fundamental

    # Theme categorization
    category = Column(String(50), index=True)  # technology, healthcare, macro, sector, etc.
    is_emerging = Column(Boolean, default=True)  # New/emerging vs established

    # Discovery tracking
    first_seen_at = Column(DateTime(timezone=True))
    last_seen_at = Column(DateTime(timezone=True))
    discovery_source = Column(String(20))  # llm_extraction, correlation_clustering, manual

    # Status
    is_active = Column(Boolean, default=True, index=True)  # Still being tracked
    is_validated = Column(Boolean, default=False)  # Validated by price correlation
    lifecycle_state = Column(String(20), nullable=False, default="candidate", index=True)
    lifecycle_state_updated_at = Column(DateTime(timezone=True))
    lifecycle_state_metadata = Column(JSON)
    candidate_since_at = Column(DateTime(timezone=True))
    activated_at = Column(DateTime(timezone=True))
    dormant_at = Column(DateTime(timezone=True))
    reactivated_at = Column(DateTime(timezone=True))
    retired_at = Column(DateTime(timezone=True))

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("pipeline", "canonical_key", name="uix_theme_clusters_pipeline_canonical_key"),
        CheckConstraint(
            "lifecycle_state IN ('candidate', 'active', 'dormant', 'reactivated', 'retired')",
            name="ck_theme_clusters_lifecycle_state",
        ),
    )


class ThemeLifecycleTransition(Base):
    """Audit trail for lifecycle-state transitions."""

    __tablename__ = "theme_lifecycle_transitions"

    id = Column(Integer, primary_key=True, index=True)
    theme_cluster_id = Column(Integer, ForeignKey("theme_clusters.id", ondelete="CASCADE"), nullable=False, index=True)
    from_state = Column(String(20), nullable=False)
    to_state = Column(String(20), nullable=False, index=True)
    actor = Column(String(80), nullable=False, default="system")
    job_name = Column(String(80))
    rule_version = Column(String(40))
    reason = Column(Text)
    transition_metadata = Column(JSON)
    transitioned_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)


class ThemeRelationship(Base):
    """Non-destructive semantic relationship edges between themes."""

    __tablename__ = "theme_relationships"

    id = Column(Integer, primary_key=True, index=True)
    source_cluster_id = Column(Integer, ForeignKey("theme_clusters.id", ondelete="CASCADE"), nullable=False, index=True)
    target_cluster_id = Column(Integer, ForeignKey("theme_clusters.id", ondelete="CASCADE"), nullable=False, index=True)
    pipeline = Column(String(20), nullable=False, index=True)
    relationship_type = Column(String(20), nullable=False, index=True)  # subset, related, distinct
    confidence = Column(Float, nullable=False, default=0.5)
    provenance = Column(String(40), nullable=False, default="rule_inference")
    evidence = Column(JSON)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "relationship_type IN ('subset', 'related', 'distinct')",
            name="ck_theme_relationship_type",
        ),
        CheckConstraint("source_cluster_id != target_cluster_id", name="ck_theme_relationship_not_self"),
        UniqueConstraint(
            "source_cluster_id",
            "target_cluster_id",
            "relationship_type",
            "pipeline",
            name="uix_theme_relationship_edge",
        ),
        Index("idx_theme_relationship_source_active", "source_cluster_id", "is_active"),
        Index("idx_theme_relationship_target_active", "target_cluster_id", "is_active"),
    )


class ThemeConstituent(Base):
    """Stocks that belong to a theme (ticker-to-theme mapping)"""

    __tablename__ = "theme_constituents"

    id = Column(Integer, primary_key=True, index=True)

    theme_cluster_id = Column(Integer, nullable=False, index=True)  # FK to theme_clusters
    symbol = Column(String(10), nullable=False, index=True)

    # How this stock was added to theme
    source = Column(String(20))  # llm_extraction, correlation, manual
    confidence = Column(Float, default=0.5)  # 0-1, confidence this stock belongs to theme

    # Mention tracking
    mention_count = Column(Integer, default=1)  # How many times mentioned with theme
    first_mentioned_at = Column(DateTime(timezone=True))
    last_mentioned_at = Column(DateTime(timezone=True))

    # Correlation validation
    correlation_to_theme = Column(Float)  # Rolling correlation to theme basket
    correlation_updated_at = Column(DateTime(timezone=True))

    # Status
    is_active = Column(Boolean, default=True)  # Still part of theme

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("theme_cluster_id", "symbol", name="uix_theme_symbol"),
        Index("idx_theme_constituents", "theme_cluster_id", "symbol"),
    )


class ThemeAlias(Base):
    """Alias records that map raw theme text to canonical clusters."""

    __tablename__ = "theme_aliases"

    id = Column(Integer, primary_key=True, index=True)
    theme_cluster_id = Column(Integer, ForeignKey("theme_clusters.id", ondelete="CASCADE"), nullable=False, index=True)
    pipeline = Column(String(20), nullable=False, index=True)

    alias_text = Column(String(200), nullable=False)
    alias_key = Column(String(96), nullable=False, index=True)

    source = Column(String(30), nullable=False, default="llm_extraction")
    confidence = Column(Float, nullable=False, default=0.5)
    evidence_count = Column(Integer, nullable=False, default=1)

    first_seen_at = Column(DateTime(timezone=True))
    last_seen_at = Column(DateTime(timezone=True))

    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("pipeline", "alias_key", name="uix_theme_alias_pipeline_alias_key"),
        Index("idx_theme_alias_cluster_active", "theme_cluster_id", "is_active"),
        Index("idx_theme_alias_source_confidence", "source", "confidence"),
    )


class ThemeMetrics(Base):
    """Daily metrics for each theme (for ranking and tracking)"""

    __tablename__ = "theme_metrics"

    id = Column(Integer, primary_key=True, index=True)

    theme_cluster_id = Column(Integer, nullable=False, index=True)  # FK to theme_clusters
    date = Column(Date, nullable=False, index=True)

    # Pipeline - which pipeline calculated these metrics
    pipeline = Column(String(20))  # technical or fundamental

    # Mention velocity (social/news signals)
    mentions_1d = Column(Integer, default=0)
    mentions_7d = Column(Integer, default=0)
    mentions_30d = Column(Integer, default=0)
    mention_velocity = Column(Float)  # 7d/30d ratio (>1 = accelerating)
    sentiment_score = Column(Float)  # -1 to 1 (bearish to bullish)

    # Price-based metrics (theme basket)
    basket_return_1d = Column(Float)  # Equal-weight basket return
    basket_return_1w = Column(Float)
    basket_return_1m = Column(Float)
    basket_rs_vs_spy = Column(Float)  # Relative strength vs SPY (0-100)

    # Correlation metrics
    avg_internal_correlation = Column(Float)  # Avg pairwise corr within theme
    correlation_tightness = Column(Float)  # Std dev of internal correlations (lower = tighter)

    # Breadth metrics
    num_constituents = Column(Integer, default=0)
    pct_above_50ma = Column(Float)  # % of stocks above 50-day MA
    pct_above_200ma = Column(Float)
    pct_positive_1w = Column(Float)  # % of stocks up on the week

    # Screener integration
    num_passing_minervini = Column(Integer, default=0)
    num_stage_2 = Column(Integer, default=0)
    avg_rs_rating = Column(Float)

    # Composite score (for ranking)
    momentum_score = Column(Float)  # Composite of velocity + RS + breadth
    rank = Column(Integer)  # 1 = top theme

    # Classification
    status = Column(String(20))  # emerging, trending, fading, dormant

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("theme_cluster_id", "date", name="uix_theme_metrics_date"),
        Index("idx_theme_metrics_date", "theme_cluster_id", "date"),
        Index("idx_theme_rank", "date", "rank"),
        Index("idx_theme_metrics_pipeline_date", "pipeline", "date"),
    )


class ThemeAlert(Base):
    """Alerts for theme events (new themes, breakouts, etc.)"""

    __tablename__ = "theme_alerts"

    id = Column(Integer, primary_key=True, index=True)

    theme_cluster_id = Column(Integer, index=True)  # FK to theme_clusters (optional)

    # Alert type
    alert_type = Column(String(30), nullable=False, index=True)
    # Types: new_theme, velocity_spike, breakout, new_constituent,
    #        correlation_spike, theme_confirmed, theme_fading

    # Alert content
    title = Column(String(200), nullable=False)
    description = Column(Text)
    severity = Column(String(10), default="info")  # info, warning, critical

    # Related data
    related_tickers = Column(JSON)  # List of relevant tickers
    metrics = Column(JSON)  # Relevant metrics at time of alert

    # Status
    is_read = Column(Boolean, default=False)
    is_dismissed = Column(Boolean, default=False)

    # Timestamps
    triggered_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    read_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_alert_unread", "is_read", "triggered_at"),
    )


class ThemePipelineRun(Base):
    """Track theme discovery pipeline executions for async processing"""

    __tablename__ = "theme_pipeline_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), nullable=False, unique=True, index=True)  # UUID
    task_id = Column(String(100), nullable=True)  # Celery task ID for progress polling

    # Pipeline results summary
    total_sources = Column(Integer, default=0)
    items_ingested = Column(Integer, default=0)
    items_processed = Column(Integer, default=0)
    items_reprocessed = Column(Integer, default=0)
    themes_extracted = Column(Integer, default=0)
    themes_updated = Column(Integer, default=0)
    alerts_generated = Column(Integer, default=0)

    # Current step tracking (for progress UI)
    current_step = Column(String(50))  # ingestion, extraction, metrics, alerts, completed

    # Status
    status = Column(String(20), default="queued")  # queued, running, completed, failed
    error_message = Column(Text, nullable=True)

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)


class ThemeEmbedding(Base):
    """Vector embeddings for theme clusters for semantic similarity search"""

    __tablename__ = "theme_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    theme_cluster_id = Column(Integer, nullable=False, unique=True, index=True)

    # Embedding data
    embedding = Column(Text, nullable=False)  # JSON-serialized numpy array
    embedding_model = Column(String(50), default="all-MiniLM-L6-v2")
    embedding_text = Column(Text)  # The text that was embedded (for debugging)
    content_hash = Column(String(64), index=True)  # SHA-256 of theme identity/content used to build embedding
    model_version = Column(String(40), default="embedding-v1", index=True)  # Embedding policy/version marker
    is_stale = Column(Boolean, default=False, nullable=False, index=True)  # Explicit stale flag for deferred refresh

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ThemeMergeSuggestion(Base):
    """Queue of theme pairs suggested for merging"""

    __tablename__ = "theme_merge_suggestions"

    id = Column(Integer, primary_key=True, index=True)

    # Theme pair
    source_cluster_id = Column(Integer, nullable=False, index=True)  # Theme to merge FROM
    target_cluster_id = Column(Integer, nullable=False, index=True)  # Theme to merge INTO

    # Similarity scores
    embedding_similarity = Column(Float)  # Cosine similarity
    llm_confidence = Column(Float)  # LLM's confidence in merge decision
    llm_reasoning = Column(Text)  # LLM's explanation
    llm_relationship = Column(String(20))  # identical, subset, related, distinct
    suggested_canonical_name = Column(String(200))  # LLM's suggested merged name

    # Status
    status = Column(String(20), default="pending", index=True)  # pending, approved, rejected, auto_merged
    reviewed_at = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        UniqueConstraint("source_cluster_id", "target_cluster_id", name="uix_merge_suggestion_pair"),
    )


class ThemeMergeHistory(Base):
    """Audit trail for theme merges"""

    __tablename__ = "theme_merge_history"

    id = Column(Integer, primary_key=True, index=True)

    # Original IDs (may be deleted after merge)
    source_cluster_id = Column(Integer)
    source_cluster_name = Column(String(200), nullable=False)
    target_cluster_id = Column(Integer)
    target_cluster_name = Column(String(200), nullable=False)

    # Merge details
    merge_type = Column(String(20), nullable=False, index=True)  # auto, manual, llm_suggested
    embedding_similarity = Column(Float)
    llm_confidence = Column(Float)
    llm_reasoning = Column(Text)

    # Stats
    constituents_merged = Column(Integer, default=0)  # Number of stocks reassigned
    mentions_merged = Column(Integer, default=0)  # Number of mentions reassigned

    # Timestamps
    merged_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    merged_by = Column(String(50), default="system")  # 'system' or user identifier
