"""Validated source facts and deterministic batch normalization."""

import json
import re
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(max_length=100)
    quote: str = Field(min_length=1, max_length=1500)


class EventFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
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


class DevelopmentFacts(EventFacts):
    theme_ids: list[int] = Field(min_length=1, max_length=20)
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


def normalize_batch(observations, *, item_id, theme_ids, sources):
    if len(observations) > 12:
        raise ValueError("invalid_development_batch")
    result = {}
    for value in observations:
        facts = DevelopmentFacts.model_validate(value)
        if not set(facts.theme_ids).issubset(theme_ids):
            raise ValueError("invalid_development_themes")
        for citation in facts.citations:
            if citation.quote not in sources.get(citation.source_id, ""):
                raise ValueError("invalid_development_citation")
        cited = normalize(" ".join(c.quote for c in facts.citations))
        for anchor in (facts.actor, facts.object, facts.reference, facts.event_time):
            if anchor and normalize(anchor) not in cited:
                raise ValueError("unsupported_development_anchor")
        key = digest(identity(facts, item_id))
        previous = result.get(key)
        if previous is None:
            result[key] = facts
            continue
        from .theme_event_state import quantity_signature

        if previous.status != facts.status or quantity_signature(
            previous.quantities
        ) != quantity_signature(facts.quantities):
            raise ValueError("conflicting_development_batch")
        citations = {
            (c.source_id, c.quote): c for c in [*previous.citations, *facts.citations]
        }
        result[key] = previous.model_copy(
            update={
                "theme_ids": sorted(set(previous.theme_ids + facts.theme_ids)),
                "citations": list(citations.values()),
            }
        )
    return list(result.values())
