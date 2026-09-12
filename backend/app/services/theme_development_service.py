"""Conservative event identity and immutable source observations."""

import json
import re
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.theme_intelligence import (
    ThemeDevelopmentEvent,
    ThemeDevelopmentObservation,
    ThemeDevelopmentTheme,
)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(max_length=100)
    quote: str = Field(min_length=1, max_length=1500)


class DevelopmentFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_ids: list[int] = Field(min_length=1, max_length=20)
    actor: str = Field(min_length=1, max_length=200)
    action: str = Field(min_length=1, max_length=120)
    object: str = Field(min_length=1, max_length=300)
    event_time: str | None = Field(default=None, max_length=120)
    reference: str | None = Field(default=None, max_length=200)
    status: Literal[
        "rumored", "announced", "confirmed", "denied", "delayed", "cancelled", "unknown"
    ]
    summary: str = Field(min_length=1, max_length=1000)
    quantities: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=20
    )
    citations: list[Citation] = Field(min_length=1, max_length=8)


def digest(value):
    return sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def normalize(value):
    return re.sub(r"\W+", " ", value.casefold()).strip()


def identity(facts, item_id):
    key = {
        field: normalize(getattr(facts, field))
        for field in ("actor", "action", "object")
    }
    if facts.reference or facts.event_time:
        key["anchor"] = normalize(facts.reference or facts.event_time)
    else:
        # Similar ticker/theme text is insufficient to join distinct events.
        key["source"] = item_id
    return key


def classify(facts, previous):
    if not previous:
        return "new_event" if facts.reference or facts.event_time else "uncertain"
    prior = previous[-1].facts
    if facts.status != prior["status"]:
        if facts.status in {"denied", "cancelled"} or prior["status"] in {
            "denied",
            "cancelled",
        }:
            return "contradiction"
        if facts.status == "unknown" or (
            prior["status"] == "confirmed" and facts.status in {"rumored", "announced"}
        ):
            return "additional_detail"
        return "material_update"
    from app.services.theme_evaluation.translation_normalization import (
        normalize_text,
        quantity_counter,
    )

    previous_quantities = quantity_counter(
        normalize_text(" ".join(prior.get("quantities", []))).quantities
    )
    new_quantities = quantity_counter(
        normalize_text(" ".join(facts.quantities)).quantities
    )
    if previous_quantities != new_quantities:
        return (
            "material_update"
            if previous_quantities and new_quantities
            else "additional_detail"
        )
    if not new_quantities and sorted(facts.quantities) != sorted(
        prior.get("quantities", [])
    ):
        return "additional_detail"
    return "repeated_coverage"


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
    if pipeline not in {"technical", "fundamental"} or len(observations) > 12:
        raise ValueError("invalid_development_batch")
    source_urls = source_urls or {}
    parsed = [DevelopmentFacts.model_validate(value) for value in observations]
    for facts in parsed:
        if not set(facts.theme_ids).issubset(theme_ids):
            raise ValueError("invalid_development_themes")
        for citation in facts.citations:
            if citation.quote not in sources.get(citation.source_id, ""):
                raise ValueError("invalid_development_citation")
        cited = normalize(" ".join(c.quote for c in facts.citations))
        for anchor in (facts.actor, facts.object, facts.reference, facts.event_time):
            if anchor and normalize(anchor) not in cited:
                raise ValueError("unsupported_development_anchor")
    existing = (
        db.query(ThemeDevelopmentObservation)
        .filter_by(content_item_id=item.id, pipeline=pipeline, revision=revision)
        .all()
    )
    if existing:
        return existing
    results = []
    for facts in parsed:
        event_identity = identity(facts, item.id)
        key = digest(event_identity)
        event = (
            db.query(ThemeDevelopmentEvent)
            .filter_by(pipeline=pipeline, event_key=key)
            .with_for_update()
            .first()
        )
        if event is None:
            event = ThemeDevelopmentEvent(
                pipeline=pipeline, event_key=key, identity=event_identity
            )
            db.add(event)
            db.flush()
        prior = (
            db.query(ThemeDevelopmentObservation)
            .filter_by(event_id=event.id, superseded=False)
            .order_by(
                ThemeDevelopmentObservation.available_at, ThemeDevelopmentObservation.id
            )
            .all()
        )
        obs_key = digest([item.id, pipeline, revision, key])
        duplicate = next(
            (row for row in results if row.observation_key == obs_key), None
        )
        if duplicate:
            for theme_id in set(facts.theme_ids) - set(duplicate.theme_ids):
                db.add(
                    ThemeDevelopmentTheme(
                        observation_id=duplicate.id, theme_id=theme_id
                    )
                )
            duplicate.theme_ids = sorted(set(duplicate.theme_ids + facts.theme_ids))
            duplicate.citations = [
                *duplicate.citations,
                *[c.model_dump() for c in facts.citations],
            ][:16]
            continue
        row = ThemeDevelopmentObservation(
            event_id=event.id,
            content_item_id=item.id,
            pipeline=pipeline,
            revision=revision,
            observation_key=obs_key,
            theme_ids=sorted(set(facts.theme_ids)),
            facts=facts.model_dump(exclude={"citations"}),
            citations=[
                {**c.model_dump(), "url": source_urls.get(c.source_id)}
                for c in facts.citations
            ],
            classification=classify(facts, prior),
            published_at=item.published_at,
            available_at=available_at,
            superseded=False,
        )
        db.add(row)
        db.flush()
        for theme_id in row.theme_ids:
            db.add(ThemeDevelopmentTheme(observation_id=row.id, theme_id=theme_id))
        results.append(row)
    # Supersede only after every incoming observation validates; keep the old rows.
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
        old.superseded = True
    db.flush()
    return results
