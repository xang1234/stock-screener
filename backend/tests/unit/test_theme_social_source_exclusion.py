from datetime import datetime, timedelta, timezone

import pytest

from app.models.theme import ContentSource, ContentItem, ThemeMention
from app.services.content_ingestion_service import ContentIngestionService
from app.services.theme_discovery_service import ThemeDiscoveryService


@pytest.mark.parametrize("social_first", [True, False])
def test_independent_legacy_observation_grants_eligibility_once(db_session, social_first):
    from app.services.theme_evidence_eligibility_service import grant_eligibility
    from app.infra.db.models.social_signals import ContentPipelineEligibility, SocialPostSource
    now = datetime.now(timezone.utc)
    social = ContentSource(name="Social", source_type="twitter", url="https://x.com/i/lists/3001", pipelines=["technical"])
    legacy = ContentSource(name="Legacy", source_type="twitter", url="https://x.com/legacy", pipelines=["technical"])
    db_session.add_all([social, legacy])
    db_session.flush()
    item = ContentItem(source_id=social.id if social_first else legacy.id, source_type="twitter", external_id="post1", published_at=now, fetched_at=now)
    db_session.add(item)
    db_session.flush()
    db_session.add(ThemeMention(content_item_id=item.id, source_type="twitter", raw_theme="Robotics", theme_cluster_id=1, pipeline="technical", mentioned_at=now, confidence=1, sentiment="bullish"))
    if social_first:
        grant_eligibility(db_session, item.id, "technical", "social", social.id, now)
        db_session.add(SocialPostSource(content_item_id=item.id, content_source_id=social.id, observed_at=now))
    else:
        grant_eligibility(db_session, item.id, "technical", "legacy", legacy.id, now)
    db_session.commit()
    discovery = ThemeDiscoveryService(db_session, pipeline="technical")
    initial_count = 0 if social_first else 1
    assert discovery._calculate_mention_metrics_batch([1], as_of_date=now)[1]["mentions_7d"] == initial_count
    assert discovery._lifecycle_snapshot(1, now=now)["source_diversity_7d"] == initial_count
    assert discovery._count_active_ingestion_days(now - timedelta(days=1), now + timedelta(seconds=1)) == initial_count
    ingestion = ContentIngestionService(db_session)

    class FixtureProvider:
        def fetch(self, source, since):
            return [{"external_id": "post1"}]

    ingestion.fetchers["twitter"] = FixtureProvider()
    assert ingestion.fetch_source(legacy) == 0
    assert ingestion.fetch_source(legacy) == 0
    if not social_first:
        grant_eligibility(db_session, item.id, "technical", "social", social.id, now)
        db_session.add(SocialPostSource(content_item_id=item.id, content_source_id=social.id, observed_at=now))
        db_session.commit()
    assert db_session.query(ContentPipelineEligibility).count() == 2
    assert db_session.query(SocialPostSource).count() == 1
    assert discovery._calculate_mention_metrics_batch([1], as_of_date=now)[1]["mentions_7d"] == 1
    assert discovery._lifecycle_snapshot(1, now=now)["source_diversity_7d"] == 1
    # Shared Social extraction and later legacy extraction may name the same
    # cluster twice; attention still counts one canonical post.
    db_session.add(ThemeMention(content_item_id=item.id, source_type="twitter", raw_theme="Robotics alias", theme_cluster_id=1, pipeline="technical", mentioned_at=now, confidence=1, sentiment="bullish"))
    db_session.commit()
    assert discovery._calculate_mention_metrics_batch([1], as_of_date=now)[1]["mentions_7d"] == 1
    assert discovery._lifecycle_snapshot(1, now=now)["mentions_7d"] == 1


def test_canonical_content_retains_each_legacy_observing_source(db_session):
    from app.infra.db.models.social_signals import ContentPipelineEligibility
    from app.services.theme_evidence_eligibility_service import (
        grant_eligibility,
        legacy_eligibility_exists,
    )

    now = datetime.now(timezone.utc)
    first = ContentSource(
        name="First legacy source",
        source_type="news",
        url="https://example.test/first",
        pipelines=["technical"],
        is_active=True,
    )
    second = ContentSource(
        name="Second legacy source",
        source_type="news",
        url="https://example.test/second",
        pipelines=["technical"],
        is_active=True,
    )
    db_session.add_all([first, second])
    db_session.flush()
    item = ContentItem(
        source_id=first.id,
        source_type="news",
        external_id="shared-canonical-item",
        published_at=now,
        fetched_at=now,
    )
    db_session.add(item)
    db_session.flush()
    db_session.add(ThemeMention(
        content_item_id=item.id,
        source_type="news",
        raw_theme="Shared observation",
        theme_cluster_id=99,
        pipeline="technical",
        mentioned_at=now,
        confidence=1.0,
        sentiment="bullish",
    ))

    grant_eligibility(
        db_session, item.id, "technical", "legacy", first.id, now
    )
    grant_eligibility(
        db_session,
        item.id,
        "technical",
        "legacy",
        second.id,
        now + timedelta(seconds=1),
    )
    db_session.commit()

    memberships = db_session.query(ContentPipelineEligibility).filter_by(
        content_item_id=item.id,
        pipeline="technical",
        channel="legacy",
    ).all()
    assert {row.originating_source_id for row in memberships} == {
        first.id,
        second.id,
    }
    lifecycle = ThemeDiscoveryService(
        db_session, pipeline="technical"
    )._lifecycle_snapshot(99, now=now + timedelta(seconds=2))
    assert lifecycle["mentions_7d"] == 1
    assert lifecycle["source_diversity_7d"] == 2

    first.is_active = False
    db_session.commit()
    assert db_session.query(ContentItem.id).filter(
        ContentItem.id == item.id,
        legacy_eligibility_exists(
            ContentItem.id,
            "technical",
            active_only=True,
            source_ids=(second.id,),
        ),
    ).one() == (item.id,)
    lifecycle = ThemeDiscoveryService(
        db_session, pipeline="technical"
    )._lifecycle_snapshot(99, now=now + timedelta(seconds=2))
    assert lifecycle["mentions_7d"] == 1
    assert lifecycle["source_diversity_7d"] == 1


def test_social_only_item_is_skipped_by_queued_legacy_extraction(db_session):
    from app.services.theme_extraction_service import ThemeExtractionService
    from app.services.theme_evidence_eligibility_service import grant_eligibility
    now = datetime.now(timezone.utc)
    source = ContentSource(name="Social", source_type="twitter", is_active=True)
    db_session.add(source)
    db_session.flush()
    item = ContentItem(source_id=source.id, source_type="twitter", content="Must not extract", published_at=now)
    db_session.add(item)
    db_session.flush()
    grant_eligibility(db_session, item.id, "technical", "social", source.id, now)
    db_session.commit()
    service = ThemeExtractionService.__new__(ThemeExtractionService)
    service.db, service.pipeline = db_session, "technical"
    assert service._process_item_transactional(item.id) == (False, 0)
    from app.services.theme_pipeline_state_service import reconcile_source_pipeline_change
    result = reconcile_source_pipeline_change(db_session, source_id=source.id, old_pipelines=["technical"], new_pipelines=["technical", "fundamental"])
    assert result["created_pending_rows"] == 0
