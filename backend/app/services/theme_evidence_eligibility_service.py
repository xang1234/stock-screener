"""Channel provenance is independent of the first canonical content insert."""
from sqlalchemy import exists, select

from app.domain.social_signals.records import validate_utc_timestamp
from app.infra.db.models.social_signals import ContentPipelineEligibility, SocialSourceConfiguration
from app.models.theme import ContentSource


def legacy_eligibility_exists(content_item_id, pipeline, *, active_only=False, source_ids=None):
    query = select(ContentPipelineEligibility.content_item_id).where(
        ContentPipelineEligibility.content_item_id == content_item_id,
        ContentPipelineEligibility.pipeline == pipeline,
        ContentPipelineEligibility.channel == "legacy",
    )
    if active_only:
        query = query.join(ContentSource, ContentSource.id == ContentPipelineEligibility.originating_source_id).where(ContentSource.is_active.is_(True))
    if source_ids is not None:
        query = query.where(ContentPipelineEligibility.originating_source_id.in_(source_ids))
    return query.correlate_except(ContentPipelineEligibility, ContentSource).exists()


def grant_eligibility(db, content_item_id, pipeline, channel, originating_source_id, observed_at):
    """First observation wins, including under concurrent canonical deduplication.

    Caller owns the observation transaction. No automatic legacy grant is made
    from ContentItem.source_id, pipeline state, or the source's current settings.
    """
    validate_utc_timestamp(observed_at, "observed_at")
    if pipeline not in {"technical", "fundamental"} or channel not in {"legacy", "social"}:
        raise ValueError("invalid_content_eligibility")
    if channel == "legacy" and is_social_owned_source(db, originating_source_id):
        raise ValueError("social_collection_owned")
    if db.get_bind().dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    db.execute(insert(ContentPipelineEligibility).values(
        content_item_id=content_item_id, pipeline=pipeline, channel=channel,
        originating_source_id=originating_source_id, observed_at=observed_at,
    ).on_conflict_do_nothing(index_elements=["content_item_id", "pipeline", "channel"]))


def is_social_owned_source(db, source_or_id):
    source = source_or_id if isinstance(source_or_id, ContentSource) else db.get(ContentSource, source_or_id)
    if source is None:
        return False
    if db.get(SocialSourceConfiguration, source.id) is not None:
        return True
    if source.source_type != "twitter":
        return False
    # Use the same locator normalization as legacy collection, including aliases.
    # Importing this module does not load private bindings or perform reads.
    from app.services.twitter_ingestion_providers import _normalized_locator, _parse_source_locator, TwitterIngestionProviderError
    try:
        ref = _parse_source_locator(_normalized_locator(source), source)
    except TwitterIngestionProviderError:
        return False
    return ref.kind == "list" and db.query(SocialSourceConfiguration.content_source_id).filter_by(x_list_id=str(int(ref.value))).first() is not None


def legacy_sources(db, query):
    """Exclude owners in SQL, then resolve aliases before returning any work."""
    query = query.filter(~exists().where(SocialSourceConfiguration.content_source_id == ContentSource.id))
    return [source for source in query.all() if not is_social_owned_source(db, source)]
