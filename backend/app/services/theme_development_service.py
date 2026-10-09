"""Persist source-bound developments through one channel-neutral event authority."""

from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.domain.economic_taxonomy.contracts import DevelopmentSupport
from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
)
from app.models.economic_taxonomy_runtime import (
    DevelopmentSelectionRevision,
    SourceFamily,
    TaxonomyAuthority,
    TaxonomySourceRevisionLog,
)
from app.models.theme import ContentItem
from app.models.theme_intelligence import (
    EconomicThemeDevelopment,
    LegacyDevelopmentEventMapping,
    ThemeDevelopmentEvent,
    ThemeDevelopmentObservation,
    ThemeDevelopmentTheme,
)
from app.services.economic_taxonomy_fence import producer_write

from .theme_development_facts import (
    DevelopmentFacts as DevelopmentFacts,  # noqa: PLC0414 -- Existing public import.
)
from .theme_development_facts import EventFacts, digest, identity, normalize_batch
from .theme_event_state import EventState
from .theme_event_state import (
    classify as classify,  # noqa: PLC0414 -- Existing public import.
)

ANALYSIS_CHANNELS = {"technical", "fundamental", "narrative"}
DEVELOPMENT_SUPPORTS = {support.value for support in DevelopmentSupport}


def current_development_event(db, event_id: int) -> ThemeDevelopmentEvent:
    """Resolve a legacy event identity without rewriting its historical rows."""

    mapping = db.scalar(
        select(LegacyDevelopmentEventMapping).where(
            LegacyDevelopmentEventMapping.old_event_id == event_id
        )
    )
    resolved_id = mapping.canonical_event_id if mapping is not None else event_id
    event = db.get(ThemeDevelopmentEvent, resolved_id)
    if event is None:
        raise KeyError(f"development event {event_id} not found")
    return event


def record_developments(
    db,
    *,
    item,
    pipeline,
    revision,
    theme_ids,
    sources,
    observations,
    available_at,
    source_urls=None,
    analysis_channel=None,
    development_support="present",
    economic_theme_ids=(),
    source_family_id: UUID | None = None,
    prepared_observations=None,
    authority_fenced=False,
    authority_epoch=None,
    economic_link_origin="economic_native",
    economic_theme_refs=None,
):
    """Persist one evidence revision and its exact development selections.

    ``economic_theme_refs`` maps the integer theme refs the model returned to
    economic theme ids (#513): each event then links to its own economic themes
    and no legacy theme link is written.

    Model/provider work is completed by the caller. Validation and deterministic
    normalization happen before the short authority-fenced write section.
    """

    channel = analysis_channel or pipeline
    if channel not in ANALYSIS_CHANNELS:
        raise ValueError("invalid_development_analysis_channel")
    if development_support not in DEVELOPMENT_SUPPORTS:
        raise ValueError("invalid_development_support")
    if economic_link_origin not in {
        "economic_native",
        "legacy_mapping",
        "compatibility",
    }:
        raise ValueError("invalid_economic_development_link_origin")
    legacy_theme_ids = sorted(set(theme_ids))
    economic_theme_refs = dict(economic_theme_refs or {})
    if economic_theme_refs and legacy_theme_ids:
        raise ValueError("development_theme_refs_conflict")
    economic_ids = sorted(
        set(economic_theme_ids) | set(economic_theme_refs.values()), key=str
    )
    source_urls = source_urls or {}
    parsed = prepared_observations or normalize_batch(
        observations,
        item_id=item.id,
        theme_ids=set(economic_theme_refs) or set(legacy_theme_ids),
        sources=sources,
        allow_empty_theme_ids=bool(economic_ids) and not economic_theme_refs,
    )
    if parsed and not legacy_theme_ids and not economic_ids:
        raise ValueError("invalid_development_themes")
    source_identity = str(source_family_id) if source_family_id else item.id

    def persist(locked_epoch):
        results, changed, selection_ids = _record_developments_fenced(
            db,
            item=item,
            pipeline=pipeline,
            analysis_channel=channel,
            revision=revision,
            legacy_theme_ids=legacy_theme_ids,
            economic_theme_ids=economic_ids,
            source_family_id=source_family_id,
            source_identity=source_identity,
            parsed=parsed,
            source_urls=source_urls,
            available_at=available_at,
            development_support=development_support,
            economic_link_origin=economic_link_origin,
            economic_theme_refs=economic_theme_refs,
        )
        if changed:
            logical_source_key = (
                f"source-family:{source_family_id}"
                if source_family_id
                else f"content-item:{item.id}"
            )
            _append_dirty_revision(
                db,
                authority_epoch=locked_epoch,
                logical_source_key=logical_source_key,
                revision_kind="development_observation",
                content_hash=digest(
                    {
                        "analysis_channel": channel,
                        "revision": revision,
                        "observation_ids": sorted(row.id for row in results),
                        "selection_revision_ids": sorted(map(str, selection_ids)),
                    }
                ),
            )
        return results

    if authority_fenced:
        if authority_epoch is None:
            raise ValueError("development_authority_epoch_required")
        return persist(authority_epoch)

    authority = db.get(TaxonomyAuthority, 1)
    expected_epoch = authority.authority_epoch if authority is not None else 1
    with producer_write(
        db,
        expected_epoch=expected_epoch,
        allowed_modes={"legacy", "shadow", "dual", "economic"},
    ) as locked_authority:
        return persist(locked_authority.authority_epoch)


def _record_developments_fenced(
    db,
    *,
    item,
    pipeline,
    analysis_channel,
    revision,
    legacy_theme_ids,
    economic_theme_ids,
    source_family_id,
    source_identity,
    parsed,
    source_urls,
    available_at,
    development_support,
    economic_link_origin,
    economic_theme_refs=None,
):
    if source_family_id is not None:
        source = db.scalar(
            select(SourceFamily)
            .where(SourceFamily.id == source_family_id)
            .with_for_update()
        )
        if source is None:
            raise ValueError("development_source_family_missing")
    else:
        source = db.scalar(
            select(ContentItem).where(ContentItem.id == item.id).with_for_update()
        )
        if source is None:
            raise ValueError("development_source_missing")
    source_filter = (
        ThemeDevelopmentObservation.source_family_id == source_family_id
        if source_family_id is not None
        else ThemeDevelopmentObservation.content_item_id == item.id
    )
    existing = db.scalars(
        select(ThemeDevelopmentObservation).where(
            source_filter,
            ThemeDevelopmentObservation.analysis_channel == analysis_channel,
            ThemeDevelopmentObservation.revision == revision,
        )
    ).all()
    changed = False
    results = list(existing)
    affected_canonical_ids: set[int] = set()

    if existing and all(not row.superseded for row in existing):
        for row in existing:
            # Per-event economic links were written with the observation; the
            # same revision re-recorded has nothing item-wide to add.
            if not economic_theme_refs:
                changed |= _ensure_links(
                    db,
                    row,
                    legacy_theme_ids=legacy_theme_ids,
                    economic_theme_ids=economic_theme_ids,
                    economic_link_origin=economic_link_origin,
                )
            affected_canonical_ids.add(current_development_event(db, row.event_id).id)
    else:
        for row in existing:
            if row.superseded:
                row.superseded = False
                changed = True
        for facts in sorted(
            parsed, key=lambda value: digest(identity(value, source_identity))
        ):
            event_identity = identity(facts, source_identity)
            key = digest(event_identity)
            event = _get_or_create_canonical_event(
                db,
                event_key=key,
                event_identity=event_identity,
                analysis_channel=analysis_channel,
            )
            affected_canonical_ids.add(event.id)
            observation_key = digest(
                [str(source_identity), analysis_channel, revision, key]
            )
            row = db.scalar(
                select(ThemeDevelopmentObservation).where(
                    ThemeDevelopmentObservation.observation_key == observation_key
                )
            )
            if row is None:
                row = ThemeDevelopmentObservation(
                    event_id=event.id,
                    content_item_id=item.id,
                    pipeline=pipeline,
                    analysis_channel=analysis_channel,
                    development_support=development_support,
                    source_family_id=source_family_id,
                    revision=revision,
                    observation_key=observation_key,
                    facts=facts.model_dump(exclude={"citations", "theme_ids"}),
                    citations=[
                        {
                            **citation.model_dump(),
                            "url": source_urls.get(citation.source_id),
                        }
                        for citation in facts.citations
                    ],
                    classification="uncertain",
                    published_at=item.published_at,
                    available_at=available_at,
                    superseded=False,
                )
                db.add(row)
                db.flush()
                changed = True
            elif row.superseded:
                row.superseded = False
                changed = True
            if economic_theme_refs:
                changed |= _ensure_links(
                    db,
                    row,
                    legacy_theme_ids=(),
                    economic_theme_ids=sorted(
                        {economic_theme_refs[ref] for ref in facts.theme_ids}, key=str
                    ),
                    economic_link_origin=economic_link_origin,
                )
            else:
                changed |= _ensure_links(
                    db,
                    row,
                    legacy_theme_ids=facts.theme_ids or legacy_theme_ids,
                    economic_theme_ids=economic_theme_ids,
                    economic_link_origin=economic_link_origin,
                )
            if row not in results:
                results.append(row)

    old_rows = db.scalars(
        select(ThemeDevelopmentObservation).where(
            source_filter,
            ThemeDevelopmentObservation.analysis_channel == analysis_channel,
            ThemeDevelopmentObservation.revision != revision,
            ThemeDevelopmentObservation.superseded.is_(False),
        )
    ).all()
    for old in old_rows:
        affected_canonical_ids.add(current_development_event(db, old.event_id).id)
        old.superseded = True
        changed = True
    db.flush()

    selection_ids = []
    for canonical_event_id in sorted(affected_canonical_ids):
        _reclassify_event_family(db, canonical_event_id)
        selection, selection_changed = _append_selection_revision(
            db, canonical_event_id
        )
        selection_ids.append(selection.id)
        changed |= selection_changed
    db.flush()
    return results, changed, selection_ids


def _ensure_links(
    db,
    observation,
    *,
    legacy_theme_ids,
    economic_theme_ids,
    economic_link_origin,
) -> bool:
    changed = False
    current_legacy = {link.theme_id for link in observation.theme_links}
    for theme_id in legacy_theme_ids:
        if theme_id not in current_legacy:
            db.add(
                ThemeDevelopmentTheme(
                    observation_id=observation.id,
                    theme_id=theme_id,
                )
            )
            changed = True
    current_economic = {
        link.economic_theme_id for link in observation.economic_theme_links
    }
    for economic_theme_id in economic_theme_ids:
        if economic_theme_id not in current_economic:
            db.add(
                EconomicThemeDevelopment(
                    observation_id=observation.id,
                    economic_theme_id=economic_theme_id,
                    link_origin=economic_link_origin,
                )
            )
            changed = True
    if changed:
        db.flush()
        db.expire(observation, ["theme_links", "economic_theme_links"])
    return changed


def _get_or_create_canonical_event(
    db,
    *,
    event_key,
    event_identity,
    analysis_channel,
) -> ThemeDevelopmentEvent:
    event = db.scalar(
        select(ThemeDevelopmentEvent)
        .where(ThemeDevelopmentEvent.canonical_event_key == event_key)
        .with_for_update()
    )
    if event is not None:
        return event
    legacy = db.scalars(
        select(ThemeDevelopmentEvent)
        .where(ThemeDevelopmentEvent.event_key == event_key)
        .order_by(ThemeDevelopmentEvent.id)
    ).all()
    if legacy:
        return _canonicalize_legacy_group(
            db,
            event_key=event_key,
            migration_run_id=uuid4(),
        )[0]
    try:
        with db.begin_nested():
            event = ThemeDevelopmentEvent(
                pipeline=analysis_channel,
                event_key=event_key,
                canonical_event_key=event_key,
                development_identity=uuid4(),
                identity=event_identity,
            )
            db.add(event)
            db.flush()
            return event
    except IntegrityError:
        return db.scalar(
            select(ThemeDevelopmentEvent)
            .where(ThemeDevelopmentEvent.canonical_event_key == event_key)
            .with_for_update()
        )


def _canonicalize_legacy_group(db, *, event_key, migration_run_id):
    rows = db.scalars(
        select(ThemeDevelopmentEvent)
        .where(ThemeDevelopmentEvent.event_key == event_key)
        .order_by(ThemeDevelopmentEvent.id)
        .with_for_update()
    ).all()
    if not rows:
        raise KeyError(f"development event key {event_key} not found")
    legacy_ids = {row.id for row in rows if row.canonical_event_key is None}
    canonical = next(
        (row for row in rows if row.canonical_event_key == event_key), rows[0]
    )
    if canonical.canonical_event_key is None:
        canonical.canonical_event_key = event_key
    if canonical.development_identity is None:
        canonical.development_identity = uuid4()
    created = 0
    for row in rows:
        if row.id not in legacy_ids:
            continue
        existing = db.scalar(
            select(LegacyDevelopmentEventMapping).where(
                LegacyDevelopmentEventMapping.old_event_id == row.id
            )
        )
        if existing is None:
            db.add(
                LegacyDevelopmentEventMapping(
                    old_event_id=row.id,
                    canonical_event_id=canonical.id,
                    old_pipeline=row.pipeline,
                    migration_run_id=migration_run_id,
                )
            )
            created += 1
        elif existing.canonical_event_id != canonical.id:
            raise ValueError("legacy_development_mapping_conflict")
    db.flush()
    return canonical, created


def _event_family_ids(db, canonical_event_id):
    mapped = set(
        db.scalars(
            select(LegacyDevelopmentEventMapping.old_event_id).where(
                LegacyDevelopmentEventMapping.canonical_event_id
                == canonical_event_id
            )
        )
    )
    mapped.add(canonical_event_id)
    return mapped


def _reclassify_event_family(db, canonical_event_id):
    state = EventState()
    previous_source = {}
    history = db.scalars(
        select(ThemeDevelopmentObservation)
        .where(
            ThemeDevelopmentObservation.event_id.in_(
                _event_family_ids(db, canonical_event_id)
            )
        )
        .order_by(
            ThemeDevelopmentObservation.available_at,
            ThemeDevelopmentObservation.id,
        )
    ).all()
    for row in history:
        facts = EventFacts.model_validate(row.facts)
        if not row.superseded:
            row.classification = state.classify(
                facts, correction=previous_source.get(row.content_item_id)
            )
            state.observe(row.facts)
        previous_source[row.content_item_id] = facts


def _append_selection_revision(db, canonical_event_id):
    event = db.scalar(
        select(ThemeDevelopmentEvent)
        .where(ThemeDevelopmentEvent.id == canonical_event_id)
        .with_for_update()
    )
    if event is None or event.development_identity is None:
        raise ValueError("canonical_development_identity_missing")
    current = db.scalars(
        select(ThemeDevelopmentObservation)
        .where(
            ThemeDevelopmentObservation.event_id.in_(
                _event_family_ids(db, canonical_event_id)
            ),
            ThemeDevelopmentObservation.superseded.is_(False),
        )
        .order_by(ThemeDevelopmentObservation.id)
    ).all()
    payload = {
        "canonical_event_id": canonical_event_id,
        "observation_ids": [
            row.id for row in current if row.development_support == "present"
        ],
        "observations": [
            {
                "id": row.id,
                "analysis_channel": row.analysis_channel,
                "development_support": row.development_support,
                "revision": row.revision,
            }
            for row in current
        ],
    }
    selected = bool(payload["observation_ids"])
    latest = db.scalar(
        select(DevelopmentSelectionRevision)
        .where(
            DevelopmentSelectionRevision.development_identity
            == event.development_identity
        )
        .order_by(DevelopmentSelectionRevision.revision_number.desc())
        .limit(1)
    )
    if latest is not None and latest.selected == selected and latest.payload == payload:
        return latest, False
    revision = DevelopmentSelectionRevision(
        development_identity=event.development_identity,
        revision_number=(latest.revision_number if latest else 0) + 1,
        selected=selected,
        payload=payload,
    )
    db.add(revision)
    db.flush()
    return revision, True


def _append_dirty_revision(
    db,
    *,
    authority_epoch,
    logical_source_key,
    revision_kind,
    content_hash,
):
    current = db.scalar(
        select(func.max(TaxonomySourceRevisionLog.revision_number)).where(
            TaxonomySourceRevisionLog.producer_kind == "development",
            TaxonomySourceRevisionLog.logical_source_key == logical_source_key,
            TaxonomySourceRevisionLog.revision_kind == revision_kind,
        )
    )
    return EconomicTaxonomyPublicationRepository(db).append_source_revision(
        producer_kind="development",
        logical_source_key=logical_source_key,
        revision_kind=revision_kind,
        revision_number=int(current or 0) + 1,
        content_hash=content_hash,
        authority_epoch=authority_epoch,
    )
