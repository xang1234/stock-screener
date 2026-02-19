"""Tests for theme extraction reprocessing: bug fix, retry logic, and silent failure detection."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import (
    ContentItem,
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

        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db_session
        service.pipeline = "technical"
        service.provider = "litellm"
        service.max_age_days = 30  # 30-day cutoff excludes 60-day-old item

        with patch.object(service, 'process_batch') as mock_batch:
            result = service.reprocess_failed_items(limit=100)

        assert result["reprocessed_count"] == 0
        mock_batch.assert_not_called()


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

        # Item that was genuinely processed with mentions (should NOT be reset)
        legit_item = _make_content_item(
            db_session, pipeline_source,
            title="Legitimate extraction",
            is_processed=True,
            processed_at=datetime.utcnow() - timedelta(hours=3),
            extraction_error=None,
            external_id="legit-1",
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

        # Verify silent_fail was reset
        db_session.refresh(silent_fail)
        assert silent_fail.is_processed is False
        assert silent_fail.processed_at is None

        # Verify legit_item was NOT reset
        db_session.refresh(legit_item)
        assert legit_item.is_processed is True
