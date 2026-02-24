"""Tests for theme extraction reprocessing: bug fix, retry logic, and silent failure detection."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ContentSource,
    ThemeAlias,
    ThemeEmbedding,
    ThemeMention,
    ThemeCluster,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def pipeline_source(db_session):
    """Create an active content source assigned to 'technical' pipeline."""
    source = ContentSource(
        name="Test Substack",
        source_type="substack",
        url="https://test.substack.com/feed",
        is_active=True,
        priority=50,
        pipelines=["technical", "fundamental"],
    )
    db_session.add(source)
    db_session.commit()
    return source


def _make_content_item(db_session, source, **overrides):
    """Helper to create a ContentItem with sensible defaults."""
    defaults = dict(
        source_id=source.id,
        source_type=source.source_type,
        source_name=source.name,
        title="Test Article",
        content="AI infrastructure is accelerating...",
        published_at=datetime.utcnow() - timedelta(days=1),
        is_processed=False,
    )
    defaults.update(overrides)
    item = ContentItem(**defaults)
    db_session.add(item)
    db_session.commit()
    return item


def _set_pipeline_state(db_session, item_id: int, pipeline: str, status: str, **overrides):
    """Helper to upsert pipeline state for a content item."""
    state = db_session.query(ContentItemPipelineState).filter(
        ContentItemPipelineState.content_item_id == item_id,
        ContentItemPipelineState.pipeline == pipeline,
    ).first()
    if not state:
        state = ContentItemPipelineState(
            content_item_id=item_id,
            pipeline=pipeline,
            status=status,
            attempt_count=overrides.pop("attempt_count", 0),
        )
        db_session.add(state)
    else:
        state.status = status
    for key, value in overrides.items():
        setattr(state, key, value)
    db_session.commit()
    return state


class TestExtractFromContentBugFix:
    """Verify that extract_from_content() re-raises non-JSON errors."""

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_rate_limit_error_reraises(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """Rate limit errors should propagate (not return [])."""
        from app.services.theme_extraction_service import ThemeExtractionService

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.llm = MagicMock()
        service.gemini_client = None
        service.configured_model = None
        service.pipeline_config = None
        service._valid_tickers = set()
        service._last_request_time = 0
        service._min_request_interval = 0
        service.max_age_days = 30
        service.theme_policy_overrides = {}
        service.ticker_pattern = __import__("re").compile(r'^[A-Z]{1,5}$')

        item = _make_content_item(db_session, pipeline_source)

        # Simulate a rate limit error from LLM
        with patch.object(service, '_try_generate_litellm', side_effect=Exception("429 Rate limit exceeded")):
            with pytest.raises(Exception, match="429 Rate limit"):
                service.extract_from_content(item)

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_json_decode_error_reraises_parse_failure(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """JSON parse errors should raise a retryable parse-failure exception."""
        from app.services.theme_extraction_service import ThemeExtractionParseError, ThemeExtractionService

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.llm = MagicMock()
        service.gemini_client = None
        service.configured_model = None
        service.pipeline_config = None
        service._valid_tickers = set()
        service._last_request_time = 0
        service._min_request_interval = 0
        service.max_age_days = 30
        service.theme_policy_overrides = {}
        service.ticker_pattern = __import__("re").compile(r'^[A-Z]{1,5}$')

        item = _make_content_item(db_session, pipeline_source)

        # LLM returns invalid JSON
        with patch.object(service, '_try_generate_litellm', return_value="This is not JSON at all"):
            with pytest.raises(ThemeExtractionParseError, match="Failed to parse LLM response"):
                service.extract_from_content(item)


class TestParseFailurePipelineSemantics:
    """Verify parse failures become explicit retryable pipeline-state errors."""

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_parse_failure_marks_retryable_state_with_error_code(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionParseError, ThemeExtractionService

        item = _make_content_item(db_session, pipeline_source, external_id="parse-failure-1")
        _set_pipeline_state(db_session, item.id, "technical", "pending", attempt_count=0)

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        with patch.object(service, "_extract_and_store_mentions", side_effect=ThemeExtractionParseError("bad json")):
            with pytest.raises(ThemeExtractionParseError, match="bad json"):
                service.process_content_item(item)

        state = db_session.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == item.id,
            ContentItemPipelineState.pipeline == "technical",
        ).first()
        assert state is not None
        assert state.status == "failed_retryable"
        assert state.attempt_count == 1
        assert state.error_code == "llm_response_parse_error"
        db_session.refresh(item)
        assert item.is_processed is False
        assert item.processed_at is None
        assert "bad json" in (item.extraction_error or "")

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_parse_failure_with_char_413_is_retryable_not_terminal(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionParseError, ThemeExtractionService

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        error = ThemeExtractionParseError("Failed to parse LLM response: Expecting value: line 1 column 414 (char 413)")
        assert service._classify_failure_status(error) == "failed_retryable"


class TestReprocessFailedItems:
    """Verify reprocess_failed_items() finds, resets, and delegates retry."""

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_reprocess_finds_and_resets_failed_items(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """Items with extraction_error should be reset and retried."""
        from app.services.theme_extraction_service import ThemeExtractionService

        # Create a failed item
        failed_item = _make_content_item(
            db_session, pipeline_source,
            title="Failed extraction",
            is_processed=True,
            processed_at=datetime.utcnow() - timedelta(hours=2),
            extraction_error="429 Rate limit exceeded",
        )
        _set_pipeline_state(
            db_session,
            failed_item.id,
            pipeline="technical",
            status="failed_retryable",
            attempt_count=1,
            error_code="exception",
            error_message="429 Rate limit exceeded",
        )

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        # Mock process_batch to avoid LLM calls
        with patch.object(service, 'process_batch', return_value={
            "processed": 1, "total_mentions": 2, "errors": 0, "pipeline": "technical"
        }) as mock_batch:
            result = service.reprocess_failed_items(limit=100)

        assert result["reprocessed_count"] == 1
        assert result["processed"] == 1
        assert result["total_mentions"] == 2

        # Verify the item was reset before process_batch was called
        db_session.refresh(failed_item)
        # After process_batch mock, item state depends on mock — check process_batch was called
        # process_batch receives item_ids to avoid "stealing" freshly ingested items
        mock_batch.assert_called_once_with(limit=100, item_ids=[failed_item.id])

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_reprocess_returns_zero_when_no_failures(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """When no failed items exist, return zero stats without calling process_batch."""
        from app.services.theme_extraction_service import ThemeExtractionService

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        with patch.object(service, 'process_batch') as mock_batch:
            result = service.reprocess_failed_items(limit=100)

        assert result["reprocessed_count"] == 0
        assert result["processed"] == 0
        mock_batch.assert_not_called()

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_reprocess_respects_age_cutoff(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """Items older than max_age_days should be excluded."""
        from app.services.theme_extraction_service import ThemeExtractionService

        # Create an old failed item (60 days ago)
        _make_content_item(
            db_session, pipeline_source,
            title="Old failure",
            is_processed=True,
            published_at=datetime.utcnow() - timedelta(days=60),
            extraction_error="timeout",
        )
        old_item = db_session.query(ContentItem).filter(ContentItem.title == "Old failure").first()
        _set_pipeline_state(
            db_session,
            old_item.id,
            pipeline="technical",
            status="failed_retryable",
            attempt_count=1,
            error_code="timeout",
            error_message="timeout",
        )

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30  # 30-day cutoff excludes 60-day-old item
        service.theme_policy_overrides = {}

        with patch.object(service, 'process_batch') as mock_batch:
            result = service.reprocess_failed_items(limit=100)

        assert result["reprocessed_count"] == 0
        mock_batch.assert_not_called()

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_reprocess_is_pipeline_scoped(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """Retry selection should only include failures for the active pipeline."""
        from app.services.theme_extraction_service import ThemeExtractionService

        technical_item = _make_content_item(
            db_session,
            pipeline_source,
            title="Technical pipeline fail",
            is_processed=True,
            extraction_error="failed",
            external_id="tech-fail",
        )
        fundamental_item = _make_content_item(
            db_session,
            pipeline_source,
            title="Fundamental pipeline fail",
            is_processed=True,
            extraction_error="failed",
            external_id="fund-fail",
        )

        _set_pipeline_state(db_session, technical_item.id, "technical", "failed_retryable", attempt_count=1)
        _set_pipeline_state(db_session, fundamental_item.id, "fundamental", "failed_retryable", attempt_count=1)

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        with patch.object(service, "process_batch", return_value={"processed": 1, "total_mentions": 0, "errors": 0, "pipeline": "technical"}) as mock_batch:
            result = service.reprocess_failed_items(limit=100)

        assert result["reprocessed_count"] == 1
        mock_batch.assert_called_once_with(limit=100, item_ids=[technical_item.id])


class TestIdentifySilentFailures:
    """Verify identify_silent_failures() finds items with 0 mentions."""

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_finds_items_with_zero_mentions(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """Items marked processed but with no theme_mentions should be detected."""
        from app.services.theme_extraction_service import ThemeExtractionService

        # Item that was "processed" but has no mentions (silent failure)
        silent_fail = _make_content_item(
            db_session, pipeline_source,
            title="Silently failed",
            is_processed=True,
            processed_at=datetime.utcnow() - timedelta(hours=5),
            extraction_error=None,
        )
        _set_pipeline_state(
            db_session,
            silent_fail.id,
            pipeline="technical",
            status="processed",
            processed_at=datetime.utcnow() - timedelta(hours=5),
        )

        # Item that was genuinely processed with mentions (should NOT be reset)
        legit_item = _make_content_item(
            db_session, pipeline_source,
            title="Legitimate extraction",
            is_processed=True,
            processed_at=datetime.utcnow() - timedelta(hours=3),
            extraction_error=None,
            external_id="legit-1",
        )
        _set_pipeline_state(
            db_session,
            legit_item.id,
            pipeline="technical",
            status="processed",
            processed_at=datetime.utcnow() - timedelta(hours=3),
        )

        # Create a cluster and mention for the legitimate item
        cluster = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(cluster)
        db_session.flush()

        mention = ThemeMention(
            content_item_id=legit_item.id,
            theme_cluster_id=cluster.id,
            source_type="substack",
            raw_theme="AI Infrastructure",
            canonical_theme="AI Infrastructure",
            pipeline="technical",
            sentiment="bullish",
            confidence=0.9,
            mentioned_at=datetime.utcnow(),
        )
        db_session.add(mention)
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        result = service.identify_silent_failures(max_age_days=30)

        assert result["reset_count"] == 1
        assert silent_fail.id in result["items"]
        assert legit_item.id not in result["items"]

        # Verify silent_fail pipeline state was reset
        db_session.refresh(silent_fail)
        state = db_session.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == silent_fail.id,
            ContentItemPipelineState.pipeline == "technical",
        ).first()
        assert state.status == "pending"

        # Verify legit_item was NOT reset
        db_session.refresh(legit_item)
        assert legit_item.is_processed is True


class TestPipelineStateDrivenBatching:
    """Verify extraction batching eligibility is pipeline-state driven."""

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_process_batch_uses_pipeline_state_not_global_flag(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """An item should be processed when pipeline state is pending even if global flag is processed."""
        from app.services.theme_extraction_service import ThemeExtractionService

        item = _make_content_item(
            db_session,
            pipeline_source,
            title="State-driven item",
            is_processed=True,
            processed_at=datetime.utcnow() - timedelta(hours=1),
            extraction_error=None,
        )
        _set_pipeline_state(db_session, item.id, "technical", "pending", attempt_count=0)

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        with patch.object(service, "_extract_and_store_mentions", return_value=0):
            result = service.process_batch(limit=10)

        assert result["processed"] == 1
        assert result["errors"] == 0

        state = db_session.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == item.id,
            ContentItemPipelineState.pipeline == "technical",
        ).first()
        assert state.status == "processed"

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_claim_is_atomic_for_in_progress_rows(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """Once claimed, a second claim attempt should fail for the same pipeline row."""
        from app.services.theme_extraction_service import ThemeExtractionService

        item = _make_content_item(db_session, pipeline_source, external_id="claim-atomic-1")
        _set_pipeline_state(db_session, item.id, "technical", "pending", attempt_count=0)

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        assert service._claim_item_for_processing(item.id) is True
        assert service._claim_item_for_processing(item.id) is False

        state = db_session.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == item.id,
            ContentItemPipelineState.pipeline == "technical",
        ).first()
        assert state.status == "in_progress"
        assert state.attempt_count == 1


class TestCompatibilityWrites:
    """Verify compatibility field updates do not clobber cross-pipeline state."""

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_reprocess_reset_does_not_flip_global_processed_flag(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """Resetting one pipeline retry state should not force global is_processed to False."""
        from app.services.theme_extraction_service import ThemeExtractionService

        item = _make_content_item(
            db_session,
            pipeline_source,
            external_id="compat-keep-processed",
            is_processed=True,
            processed_at=datetime.utcnow() - timedelta(hours=2),
            extraction_error="technical failed",
        )
        _set_pipeline_state(db_session, item.id, "technical", "failed_retryable", attempt_count=2)
        _set_pipeline_state(db_session, item.id, "fundamental", "processed", attempt_count=1)

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        with patch.object(service, "process_batch", return_value={"processed": 0, "total_mentions": 0, "errors": 0, "pipeline": "technical"}):
            service.reprocess_failed_items(limit=10)

        db_session.refresh(item)
        assert item.is_processed is True

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_process_content_item_updates_pipeline_state(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """Direct process_content_item() should still maintain pipeline-state transitions."""
        from app.services.theme_extraction_service import ThemeExtractionService

        item = _make_content_item(db_session, pipeline_source, external_id="direct-api-state")

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        with patch.object(service, "_extract_and_store_mentions", return_value=1):
            mentions = service.process_content_item(item)

        assert mentions == 1
        state = db_session.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == item.id,
            ContentItemPipelineState.pipeline == "technical",
        ).first()
        assert state is not None
        assert state.status == "processed"


class TestThemeClusterLabelPreservation:
    """Verify ingestion does not overwrite analyst-managed cluster labels."""

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_get_or_create_cluster_preserves_existing_display_name(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        """Existing display_name should remain unchanged for analyst-managed labels."""
        from app.services.theme_extraction_service import ThemeExtractionService

        cluster = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="Analyst Preferred Label",
            name="Analyst Preferred Label",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(cluster)
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got = service._get_or_create_cluster({"theme": "A.I. Infrastructure"})
        db_session.refresh(cluster)

        assert got.id == cluster.id
        assert cluster.display_name == "Analyst Preferred Label"
        assert cluster.name == "Analyst Preferred Label"

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_get_or_create_cluster_uses_alias_key_exact_match(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        """Alias-key lookup should map a raw alias to an existing cluster without creating duplicates."""
        from app.services.theme_extraction_service import ThemeExtractionService

        cluster = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(cluster)
        db_session.flush()
        db_session.add(
            ThemeAlias(
                theme_cluster_id=cluster.id,
                pipeline="technical",
                alias_text="AI Infra",
                alias_key="ai_infra",
                source="manual",
                confidence=0.9,
                evidence_count=3,
                first_seen_at=datetime.utcnow(),
                last_seen_at=datetime.utcnow(),
                is_active=True,
            )
        )
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got = service._get_or_create_cluster({"theme": "AI Infra", "confidence": 0.8})
        cluster_count = db_session.query(ThemeCluster).count()

        assert got.id == cluster.id
        assert cluster_count == 1

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_get_or_create_cluster_reactivates_inactive_canonical_match(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        """Canonical-key fallback should reactivate matching inactive clusters instead of creating duplicates."""
        from app.services.theme_extraction_service import ThemeExtractionService

        cluster = ThemeCluster(
            canonical_key="ai_infra",
            display_name="AI Infra",
            name="AI Infra",
            pipeline="technical",
            aliases=["AI Infra"],
            is_active=False,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(cluster)
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got = service._get_or_create_cluster({"theme": "AI Infra", "confidence": 0.8})
        db_session.refresh(cluster)

        assert got.id == cluster.id
        assert cluster.is_active is True
        assert db_session.query(ThemeCluster).count() == 1

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_reactivates_inactive_canonical_with_reason(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        inactive = ThemeCluster(
            canonical_key="ai_next_wave",
            display_name="AI Next Wave",
            name="AI Next Wave",
            pipeline="technical",
            aliases=["AI Next Wave"],
            is_active=False,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(inactive)
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got_cluster, decision = service._resolve_cluster_match({"theme": "AI Next Wave", "confidence": 0.8})
        db_session.refresh(inactive)

        assert got_cluster.id == inactive.id
        assert inactive.is_active is True
        assert decision.method == "exact_canonical_key"
        assert decision.fallback_reason == "reactivated_inactive_canonical_match"
        assert db_session.query(ThemeCluster).count() == 1

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_uses_exact_display_name_stage_when_key_misses(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        cluster = ThemeCluster(
            canonical_key="quantum_computing",
            display_name="Quantum Computing",
            name="Quantum Computing",
            pipeline="technical",
            aliases=["Quantum Computing"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(cluster)
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got_cluster, decision = service._resolve_cluster_match({"theme": "Quantum tech", "confidence": 0.8})

        assert got_cluster.id == cluster.id
        assert decision.method == "exact_display_name"
        assert decision.score == 1.0
        assert decision.fallback_reason is None

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_emits_alias_decision(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        cluster = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(cluster)
        db_session.flush()
        db_session.add(
            ThemeAlias(
                theme_cluster_id=cluster.id,
                pipeline="technical",
                alias_text="AI Infra",
                alias_key="ai_infra",
                source="manual",
                confidence=0.9,
                evidence_count=3,
                first_seen_at=datetime.utcnow(),
                last_seen_at=datetime.utcnow(),
                is_active=True,
            )
        )
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got_cluster, decision = service._resolve_cluster_match({"theme": "AI Infra", "confidence": 0.8})
        assert got_cluster.id == cluster.id
        assert decision.method == "exact_alias_key"
        assert decision.score == 1.0
        assert decision.threshold_version == "match-v1"
        assert decision.selected_cluster_id == cluster.id
        assert decision.fallback_reason is None

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_blocks_low_quality_alias_auto_attach(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        legacy_cluster = ThemeCluster(
            canonical_key="legacy_ai_theme",
            display_name="Legacy AI Theme",
            name="Legacy AI Theme",
            pipeline="technical",
            aliases=["Legacy AI Theme"],
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(legacy_cluster)
        db_session.flush()
        alias_row = ThemeAlias(
            theme_cluster_id=legacy_cluster.id,
            pipeline="technical",
            alias_text="AI Neoinfra",
            alias_key="ai_neoinfra",
            source="llm_extraction",
            confidence=0.6,
            evidence_count=1,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
            is_active=True,
        )
        db_session.add(alias_row)
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got_cluster, decision = service._resolve_cluster_match({"theme": "AI Neoinfra", "confidence": 0.8})
        db_session.refresh(alias_row)

        assert got_cluster.id != legacy_cluster.id
        assert decision.method == "create_new_cluster"
        assert decision.fallback_reason == "alias_match_below_auto_attach_threshold"
        assert decision.best_alternative_cluster_id == legacy_cluster.id
        assert decision.best_alternative_score is not None
        # Blocked Stage B contributes counter-evidence to the old alias mapping.
        assert alias_row.evidence_count == 2
        assert alias_row.confidence < 0.6

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_clears_fallback_when_alias_target_inactive_but_canonical_exists(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        active_cluster = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        inactive_cluster = ThemeCluster(
            canonical_key="legacy_ai_infra",
            display_name="Legacy AI Infra",
            name="Legacy AI Infra",
            pipeline="technical",
            aliases=["Legacy AI Infra"],
            is_active=False,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add_all([active_cluster, inactive_cluster])
        db_session.flush()
        db_session.add(
            ThemeAlias(
                theme_cluster_id=inactive_cluster.id,
                pipeline="technical",
                alias_text="A.I. Infrastructure",
                alias_key="ai_infrastructure",
                source="manual",
                confidence=0.9,
                evidence_count=2,
                first_seen_at=datetime.utcnow(),
                last_seen_at=datetime.utcnow(),
                is_active=True,
            )
        )
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got_cluster, decision = service._resolve_cluster_match({"theme": "A.I. Infrastructure", "confidence": 0.8})
        assert got_cluster.id == active_cluster.id
        assert decision.method == "exact_canonical_key"
        assert decision.fallback_reason is None
        assert decision.best_alternative_cluster_id is None
        assert decision.best_alternative_score is None

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_prefers_canonical_stage_a_before_alias(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        canonical_cluster = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        alias_target_cluster = ThemeCluster(
            canonical_key="legacy_ai_infra",
            display_name="Legacy AI Infra",
            name="Legacy AI Infra",
            pipeline="technical",
            aliases=["Legacy AI Infra"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add_all([canonical_cluster, alias_target_cluster])
        db_session.flush()
        db_session.add(
            ThemeAlias(
                theme_cluster_id=alias_target_cluster.id,
                pipeline="technical",
                alias_text="A.I. Infrastructure",
                alias_key="ai_infrastructure",
                source="manual",
                confidence=0.8,
                evidence_count=2,
                first_seen_at=datetime.utcnow(),
                last_seen_at=datetime.utcnow(),
                is_active=True,
            )
        )
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got_cluster, decision = service._resolve_cluster_match({"theme": "AI Infrastructure", "confidence": 0.8})
        assert got_cluster.id == canonical_cluster.id
        assert decision.method == "exact_canonical_key"

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_uses_fuzzy_lexical_for_clear_high_score(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        target = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        distractor = ThemeCluster(
            canonical_key="quantum_computing",
            display_name="Quantum Computing",
            name="Quantum Computing",
            pipeline="technical",
            aliases=["Quantum Computing"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add_all([target, distractor])
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got_cluster, decision = service._resolve_cluster_match({"theme": "AI Infrastructur", "confidence": 0.8})
        assert got_cluster.id == target.id
        assert decision.method == "fuzzy_lexical"
        assert decision.score >= decision.threshold
        assert decision.threshold_version == "match-v1"

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_uses_ambiguous_review_mode(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        candidate_a = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        candidate_b = ThemeCluster(
            canonical_key="ai_infrastructures",
            display_name="AI Infrastructures",
            name="AI Infrastructures",
            pipeline="technical",
            aliases=["AI Infrastructures"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add_all([candidate_a, candidate_b])
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got_cluster, decision = service._resolve_cluster_match({"theme": "AI Infrastructur", "confidence": 0.8})
        assert got_cluster.id not in {candidate_a.id, candidate_b.id}
        assert decision.method == "create_new_cluster"
        assert decision.fallback_reason == "fuzzy_ambiguous_review"
        assert decision.best_alternative_cluster_id in {candidate_a.id, candidate_b.id}
        assert decision.best_alternative_score is not None

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_uses_low_confidence_review_mode(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        candidate = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(candidate)
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        got_cluster, decision = service._resolve_cluster_match(
            {"theme": "AI infrastructure cycle", "confidence": 0.8}
        )
        assert got_cluster.id != candidate.id
        assert decision.method == "create_new_cluster"
        assert decision.fallback_reason == "fuzzy_low_confidence_review"
        assert decision.best_alternative_cluster_id == candidate.id
        assert decision.best_alternative_score is not None

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_uses_embedding_similarity_for_stage_d_attach(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        class _StubEncoder:
            def encode(self, _text, convert_to_numpy=True):
                _ = convert_to_numpy
                return [1.0, 0.0]

        target = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        distractor = ThemeCluster(
            canonical_key="nuclear_energy",
            display_name="Nuclear Energy",
            name="Nuclear Energy",
            pipeline="technical",
            aliases=["Nuclear Energy"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add_all([target, distractor])
        db_session.flush()
        db_session.add_all(
            [
                ThemeEmbedding(
                    theme_cluster_id=target.id,
                    embedding="[1.0, 0.0]",
                    embedding_model="all-MiniLM-L6-v2",
                ),
                ThemeEmbedding(
                    theme_cluster_id=distractor.id,
                    embedding="[0.0, 1.0]",
                    embedding_model="all-MiniLM-L6-v2",
                ),
            ]
        )
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}
        service._embedding_encoder = None

        with patch.object(service, "_get_embedding_encoder", return_value=_StubEncoder()):
            got_cluster, decision = service._resolve_cluster_match({"theme": "Compute Fabric Demand", "confidence": 0.8})

        assert got_cluster.id == target.id
        assert decision.method == "embedding_similarity"
        assert decision.score >= decision.threshold
        assert decision.threshold_version == "embedding-v1"
        assert decision.score_model == "all-MiniLM-L6-v2"
        assert decision.score_model_version == "embedding-v1"

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_embedding_stage_falls_back_to_stale_rows_when_no_fresh_available(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        class _StubEncoder:
            def encode(self, _text, convert_to_numpy=True):
                _ = convert_to_numpy
                return [1.0, 0.0]

        candidate = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(candidate)
        db_session.flush()
        db_session.add(
            ThemeEmbedding(
                theme_cluster_id=candidate.id,
                embedding="[1.0, 0.0]",
                embedding_model="all-MiniLM-L6-v2",
                model_version="embedding-v1",
                is_stale=True,
                updated_at=datetime.utcnow(),
            )
        )
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}
        service._embedding_encoder = None

        with patch.object(service, "_get_embedding_encoder", return_value=_StubEncoder()):
            got_cluster, decision = service._resolve_cluster_match({"theme": "Compute Fabric Demand", "confidence": 0.8})

        assert got_cluster.id == candidate.id
        assert decision.method == "embedding_similarity"
        assert decision.score_model == "all-MiniLM-L6-v2"
        assert decision.score_model_version == "embedding-v1"

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_embedding_stage_skips_stale_or_model_mismatch(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        candidate = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(candidate)
        db_session.flush()
        db_session.add(
            ThemeEmbedding(
                theme_cluster_id=candidate.id,
                embedding="[1.0, 0.0]",
                embedding_model="different-model-v9",
                updated_at=datetime.utcnow() - timedelta(days=180),
            )
        )
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}
        service._embedding_encoder = None

        got_cluster, decision = service._resolve_cluster_match({"theme": "Rates and Liquidity Inflection", "confidence": 0.8})

        assert got_cluster.id != candidate.id
        assert decision.method == "create_new_cluster"
        assert decision.score_model is None
        assert decision.score_model_version is None

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_embedding_stage_sets_ambiguous_review_reason(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        class _StubEncoder:
            def encode(self, _text, convert_to_numpy=True):
                _ = convert_to_numpy
                return [1.0, 0.0]

        candidate_a = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        candidate_b = ThemeCluster(
            canonical_key="ai_datacenter_power",
            display_name="AI Datacenter Power",
            name="AI Datacenter Power",
            pipeline="technical",
            aliases=["AI Datacenter Power"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add_all([candidate_a, candidate_b])
        db_session.flush()
        db_session.add_all(
            [
                ThemeEmbedding(
                    theme_cluster_id=candidate_a.id,
                    embedding="[0.8, 0.6]",
                    embedding_model="all-MiniLM-L6-v2",
                ),
                ThemeEmbedding(
                    theme_cluster_id=candidate_b.id,
                    embedding="[0.79, 0.613]",
                    embedding_model="all-MiniLM-L6-v2",
                ),
            ]
        )
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}
        service._embedding_encoder = None

        with patch.object(service, "_get_embedding_encoder", return_value=_StubEncoder()):
            got_cluster, decision = service._resolve_cluster_match({"theme": "AI Compute Fabric Demand", "confidence": 0.8})

        assert got_cluster.id not in {candidate_a.id, candidate_b.id}
        assert decision.method == "create_new_cluster"
        assert decision.fallback_reason == "embedding_ambiguous_review"
        assert decision.best_alternative_cluster_id in {candidate_a.id, candidate_b.id}
        assert decision.threshold_version == "embedding-v1"
        assert decision.score_model == "all-MiniLM-L6-v2"
        assert decision.score_model_version == "embedding-v1"

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_resolve_cluster_match_embedding_stage_ignores_malformed_vectors(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        class _StubEncoder:
            def encode(self, _text, convert_to_numpy=True):
                _ = convert_to_numpy
                return [1.0, 0.0]

        candidate = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(candidate)
        db_session.flush()
        db_session.add(
            ThemeEmbedding(
                theme_cluster_id=candidate.id,
                embedding="[1.0, 0.0, 0.0]",
                embedding_model="all-MiniLM-L6-v2",
            )
        )
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}
        service._embedding_encoder = None

        with patch.object(service, "_get_embedding_encoder", return_value=_StubEncoder()):
            got_cluster, decision = service._resolve_cluster_match({"theme": "AI Compute Fabric Demand", "confidence": 0.8})

        assert got_cluster.id != candidate.id
        assert decision.method == "create_new_cluster"

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_extract_and_store_mentions_persists_decision_fields(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.services.theme_extraction_service import ThemeExtractionService

        item = _make_content_item(db_session, pipeline_source, source_type="news")

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        with patch.object(
            service,
            "extract_from_content",
            return_value=[
                {
                    "theme": "New Theme Idea",
                    "tickers": ["NVDA"],
                    "sentiment": "bullish",
                    "confidence": 0.8,
                    "excerpt": "example",
                }
            ],
        ):
            created = service._extract_and_store_mentions(item)

        assert created == 1
        mention = db_session.query(ThemeMention).filter(ThemeMention.content_item_id == item.id).one()
        assert mention.match_method == "create_new_cluster"
        assert mention.match_score == 0.0
        assert mention.match_threshold == 1.0
        assert mention.threshold_version == "match-v1"
        assert mention.match_score_model is None
        assert mention.match_score_model_version is None
        assert mention.match_fallback_reason == "no_existing_cluster_match"

    @patch("app.services.theme_extraction_service.ThemeExtractionService._init_client")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_configured_model")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_pipeline_config")
    @patch("app.services.theme_extraction_service.ThemeExtractionService._load_reprocessing_config")
    def test_extract_and_store_mentions_persists_embedding_model_metadata(
        self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source
    ):
        from app.domain.theme_matching import MatchDecision
        from app.services.theme_extraction_service import ThemeExtractionService

        item = _make_content_item(db_session, pipeline_source, source_type="news")
        cluster = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            aliases=["AI Infrastructure"],
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        db_session.add(cluster)
        db_session.commit()

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30
        service.theme_policy_overrides = {}

        with patch.object(
            service,
            "extract_from_content",
            return_value=[
                {
                    "theme": "AI Compute Fabric Demand",
                    "tickers": ["NVDA"],
                    "sentiment": "bullish",
                    "confidence": 0.8,
                    "excerpt": "example",
                }
            ],
        ), patch.object(
            service,
            "_resolve_cluster_match",
            return_value=(
                cluster,
                MatchDecision(
                    selected_cluster_id=cluster.id,
                    method="embedding_similarity",
                    score=0.91,
                    threshold=0.85,
                    threshold_version="embedding-v1",
                    score_model="all-MiniLM-L6-v2",
                    score_model_version="embedding-v1",
                ),
            ),
        ):
            created = service._extract_and_store_mentions(item)

        assert created == 1
        mention = db_session.query(ThemeMention).filter(ThemeMention.content_item_id == item.id).one()
        assert mention.match_method == "embedding_similarity"
        assert mention.threshold_version == "embedding-v1"
        assert mention.match_score_model == "all-MiniLM-L6-v2"
        assert mention.match_score_model_version == "embedding-v1"
