"""Current source authorization for bounded attachment queue queries."""

from sqlalchemy import and_, exists, or_, select

from app.config import settings
from app.infra.db.models.social_signals import (
    ContentPipelineEligibility,
    SocialSourceConfiguration,
    SocialSourceRegistry,
)
from app.models.theme import ContentSource


def attachment_eligibility(item_id):
    """SQL predicate; never materialize all historical attachment parent IDs."""
    social_enabled = exists(
        select(SocialSourceRegistry.id).where(
            SocialSourceRegistry.id == 1,
            SocialSourceRegistry.mode != "off",
            SocialSourceRegistry.provider != "disabled",
        )
    )
    social_source = exists(
        select(SocialSourceConfiguration.content_source_id).where(
            SocialSourceConfiguration.content_source_id == ContentSource.id,
            SocialSourceConfiguration.lifecycle_state == "enabled",
        )
    )
    social_owned = exists(
        select(SocialSourceConfiguration.content_source_id).where(
            SocialSourceConfiguration.content_source_id == ContentSource.id
        )
    )
    return exists(
        select(ContentPipelineEligibility.content_item_id)
        .join(
            ContentSource,
            ContentSource.id == ContentPipelineEligibility.originating_source_id,
        )
        .where(
            ContentPipelineEligibility.content_item_id == item_id,
            ContentSource.is_active.is_(True),
            or_(
                and_(
                    ContentPipelineEligibility.channel == "legacy",
                    bool(settings.feature_themes),
                    ~social_owned,
                ),
                and_(
                    ContentPipelineEligibility.channel == "social",
                    social_enabled,
                    social_source,
                ),
            ),
        )
    )


def authorize_attachment(db, item_id):
    """Check permission in the lease transaction, including legacy list aliases."""
    from app.services.theme_evidence_eligibility_service import is_social_owned_source

    # Use the same registry-first order as the social writer/admin services.
    registry = db.scalar(
        select(SocialSourceRegistry)
        .where(SocialSourceRegistry.id == 1)
        .with_for_update()
    )
    rows = db.execute(
        select(ContentPipelineEligibility.channel, ContentSource)
        .join(
            ContentSource,
            ContentSource.id == ContentPipelineEligibility.originating_source_id,
        )
        .where(
            ContentPipelineEligibility.content_item_id == item_id,
            ContentSource.is_active.is_(True),
        )
        .with_for_update()
    ).all()
    for channel, source in rows:
        if (
            channel == "legacy"
            and settings.feature_themes
            and not is_social_owned_source(db, source)
        ):
            return True
        if (
            channel == "social"
            and registry
            and registry.mode != "off"
            and registry.provider != "disabled"
        ):
            config = db.get(SocialSourceConfiguration, source.id)
            if config and config.lifecycle_state == "enabled":
                return True
    return False
