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
    def test_json_decode_error_returns_empty(self, mock_reproc, mock_pipeline, mock_model, mock_client, db_session, pipeline_source):
        """JSON parse errors should still return [] (LLM responded, just bad format)."""
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
        service.ticker_pattern = __import__("re").compile(r'^[A-Z]{1,5}$')

        item = _make_content_item(db_session, pipeline_source)

        # LLM returns invalid JSON
        with patch.object(service, '_try_generate_litellm', return_value="This is not JSON at all"):
            result = service.extract_from_content(item)
            assert result == []


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

        with patch.object(service, "_extract_and_store_mentions", return_value=1):
            mentions = service.process_content_item(item)

        assert mentions == 1
        state = db_session.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == item.id,
            ContentItemPipelineState.pipeline == "technical",
        ).first()
        assert state is not None
        assert state.status == "processed"
