"""Persist validated observations with a single membership authority."""

from sqlalchemy.exc import IntegrityError

from app.models.theme_intelligence import (
    ThemeDevelopmentEvent,
    ThemeDevelopmentObservation,
    ThemeDevelopmentTheme,
)

from .theme_development_facts import (
    DevelopmentFacts as DevelopmentFacts,  # noqa: PLC0414 -- Existing public import.
)
from .theme_development_facts import EventFacts, digest, identity, normalize_batch
from .theme_event_state import EventState
from .theme_event_state import (
    classify as classify,  # noqa: PLC0414 -- Existing public import.
)


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
):
    if pipeline not in {"technical", "fundamental"}:
        raise ValueError("invalid_development_batch")
    source_urls = source_urls or {}
    parsed = normalize_batch(
        observations, item_id=item.id, theme_ids=theme_ids, sources=sources
    )
    existing = (
        db.query(ThemeDevelopmentObservation)
        .filter_by(content_item_id=item.id, pipeline=pipeline, revision=revision)
        .all()
    )
    if existing and all(not row.superseded for row in existing):
        return existing
    results = existing
    for row in existing:
        row.superseded = False
    for facts in (
        []
        if existing
        else sorted(parsed, key=lambda facts: digest(identity(facts, item.id)))
    ):
        event_identity = identity(facts, item.id)
        key = digest(event_identity)
        event = (
            db.query(ThemeDevelopmentEvent)
            .filter_by(pipeline=pipeline, event_key=key)
            .with_for_update()
            .first()
        )
        if event is None:
            try:
                with db.begin_nested():
                    event = ThemeDevelopmentEvent(
                        pipeline=pipeline, event_key=key, identity=event_identity
                    )
                    db.add(event)
                    db.flush()
            except IntegrityError:
                event = (
                    db.query(ThemeDevelopmentEvent)
                    .filter_by(pipeline=pipeline, event_key=key)
                    .with_for_update()
                    .one()
                )
        obs_key = digest([item.id, pipeline, revision, key])
        row = ThemeDevelopmentObservation(
            event_id=event.id,
            content_item_id=item.id,
            pipeline=pipeline,
            revision=revision,
            observation_key=obs_key,
            facts=facts.model_dump(exclude={"citations", "theme_ids"}),
            citations=[
                {**c.model_dump(), "url": source_urls.get(c.source_id)}
                for c in facts.citations
            ],
            classification="uncertain",
            published_at=item.published_at,
            available_at=available_at,
            superseded=False,
        )
        db.add(row)
        db.flush()
        row.theme_links = [
            ThemeDevelopmentTheme(theme_id=theme_id) for theme_id in facts.theme_ids
        ]
        results.append(row)
    # Supersede only after every incoming observation validates; keep the old rows.
    affected_events = {row.event_id for row in results}
    for old in (
        db.query(ThemeDevelopmentObservation)
        .filter(
            ThemeDevelopmentObservation.content_item_id == item.id,
            ThemeDevelopmentObservation.pipeline == pipeline,
            ThemeDevelopmentObservation.revision != revision,
            ThemeDevelopmentObservation.superseded.is_(False),
        )
        .all()
    ):
        affected_events.add(old.event_id)
        old.superseded = True
    db.flush()
    for event_id in sorted(affected_events):
        state = EventState()
        history = (
            db.query(ThemeDevelopmentObservation)
            .filter_by(event_id=event_id)
            .order_by(
                ThemeDevelopmentObservation.available_at, ThemeDevelopmentObservation.id
            )
        )
        previous_source = {}
        for row in history:
            facts = EventFacts.model_validate(row.facts)
            if not row.superseded:
                row.classification = state.classify(
                    facts, correction=previous_source.get(row.content_item_id)
                )
                state.observe(row.facts)
            previous_source[row.content_item_id] = facts
    db.flush()
    return results
