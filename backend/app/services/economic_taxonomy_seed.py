"""Governed V1 Economic Taxonomy facet catalog and normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.infra.db.repositories.economic_taxonomy_repo import (
    EconomicTaxonomyRepository,
)
from app.services.economic_taxonomy_fence import exclusive_publication


@dataclass(frozen=True, slots=True)
class DimensionDefinition:
    definition: str
    inclusion_semantics: str
    exclusion_semantics: str
    value_type: str = "controlled_term"
    cardinality: str = "many"
    scope: str = "theme"
    normalization_policy: str = "economic-facet-v1"


INITIAL_DIMENSIONS = {
    "industry": DimensionDefinition(
        "The economic industry whose supply, demand, and profit pool define exposure.",
        "Include participation in the named industry's economics.",
        "Exclude a product, customer, or end market without industry participation.",
    ),
    "technology": DimensionDefinition(
        "A technology that is itself the defining economic mechanism.",
        "Include only when the technology directly defines the exposure.",
        "Exclude technology used only as a demand end market or incidental enabler.",
    ),
    "product": DimensionDefinition(
        "A specific product whose economics define the exposure.",
        "Include direct product participation supported by evidence.",
        "Exclude inferred product specificity from a broader industry claim.",
    ),
    "commodity": DimensionDefinition(
        "A physical commodity whose price, supply, or demand defines exposure.",
        "Include direct commodity economics.",
        "Exclude downstream products without material commodity sensitivity.",
    ),
    "end_market": DimensionDefinition(
        "The demand market ultimately served by the exposure.",
        "Include evidence of demand serving the named market.",
        "Exclude the market label as a defining technology or industry by itself.",
    ),
    "customer": DimensionDefinition(
        "A governed customer class or named demand counterparty.",
        "Include economically material customer exposure.",
        "Exclude incidental or ungrounded customer mentions.",
    ),
    "supply_chain": DimensionDefinition(
        "The participant's role or location in an economic supply chain.",
        "Include a supported value-chain position.",
        "Exclude unsupported adjacency.",
    ),
    "geography": DimensionDefinition(
        "The geography that materially scopes the exposure.",
        "Include economically meaningful geographic scope.",
        "Exclude source location or listing venue alone.",
    ),
    "policy": DimensionDefinition(
        "A public-policy program or direction defining the exposure.",
        "Include explicit policy-linked economic mechanisms.",
        "Exclude general political commentary.",
    ),
    "regulation": DimensionDefinition(
        "A regulatory regime or change defining the exposure.",
        "Include direct regulatory economic effects.",
        "Exclude generic compliance references.",
    ),
    "macro": DimensionDefinition(
        "A macroeconomic mechanism materially defining the exposure.",
        "Include a specific transmission mechanism.",
        "Exclude market direction or technical setups.",
    ),
    "infrastructure": DimensionDefinition(
        "A physical or digital infrastructure class defining the exposure.",
        "Include direct infrastructure buildout or operation economics.",
        "Exclude incidental infrastructure usage.",
    ),
}


INITIAL_FACET_VALUES = {
    "industry": {"memory_semiconductors": "Memory"},
    "technology": {"artificial_intelligence": "Artificial Intelligence"},
    "product": {"hbm": "HBM"},
    "end_market": {"artificial_intelligence": "AI"},
}


_VALUE_ALIASES = {
    "end_market": {
        "ai": "artificial_intelligence",
        "ai_workloads": "artificial_intelligence",
        "artificial_intelligence": "artificial_intelligence",
    },
    "industry": {
        "memory": "memory_semiconductors",
        "memory_economics": "memory_semiconductors",
        "memory_semiconductor": "memory_semiconductors",
        "memory_semiconductors": "memory_semiconductors",
    },
    "product": {
        "hbm": "hbm",
        "high_bandwidth_memory": "hbm",
    },
    "technology": {
        "ai": "artificial_intelligence",
        "artificial_intelligence": "artificial_intelligence",
    },
}


class UnknownFacetDimension(ValueError):
    pass


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("facet value must be non-empty")
    return normalized


def normalize_facet(dimension_key: str, value: str) -> str:
    """Normalize a value without moving it into a different semantic dimension."""

    key = _slug(dimension_key)
    if key not in INITIAL_DIMENSIONS:
        raise UnknownFacetDimension(key)
    normalized = _slug(value)
    return _VALUE_ALIASES.get(key, {}).get(normalized, normalized)


def seed_initial_dimensions(
    repository: EconomicTaxonomyRepository,
    taxonomy_version_id: UUID,
    *,
    actor: str = "system:economic-taxonomy-refresh",
) -> None:
    """Seed the exact governed V1 dimension catalog into a draft snapshot."""

    for key, definition in INITIAL_DIMENSIONS.items():
        repository.add_dimension(
            taxonomy_version_id,
            key=key,
            definition=definition.definition,
            inclusion_semantics=definition.inclusion_semantics,
            exclusion_semantics=definition.exclusion_semantics,
            value_type=definition.value_type,
            cardinality=definition.cardinality,
            scope=definition.scope,
            normalization_policy=definition.normalization_policy,
            actor=actor,
        )
    for dimension_key, values in INITIAL_FACET_VALUES.items():
        for normalized_value, display_value in values.items():
            repository.add_facet_value(
                taxonomy_version_id,
                dimension_key,
                normalized_value,
                display_value,
                actor=actor,
            )



class SeedRefused(RuntimeError):
    """The authority row already has a processing taxonomy or is not plain legacy."""


def seed_governed_snapshot(session: Session, *, actor: str) -> UUID:
    """Seed the governed V1 draft and make it the processing taxonomy.

    Runs under the exclusive writer fence, whose authority lock inserts the
    implicit ``legacy`` row when none exists. That row, or one a fenced legacy
    write created earlier (e.g. a theme metrics run), is adopted with its epoch
    kept (#473). Refuses a row that already has a processing taxonomy, is not
    in legacy mode, or has writes fenced; then nothing is written.
    """
    if actor.startswith("system:"):
        raise SeedRefused(f"actor {actor!r} uses the reserved 'system:' prefix")
    with exclusive_publication(session) as authority:
        if (
            authority.mode != "legacy"
            or authority.processing_taxonomy_version_id is not None
            or authority.writes_fenced
        ):
            session.rollback()
            raise SeedRefused(
                "taxonomy authority is not an unseeded legacy row: "
                f"mode={authority.mode} "
                f"processing_taxonomy_version_id={authority.processing_taxonomy_version_id} "
                f"writes_fenced={authority.writes_fenced}"
            )
        repository = EconomicTaxonomyRepository(session)
        draft = repository.create_draft(actor=actor, reason="V1 governed seed")
        seed_initial_dimensions(repository, draft.id, actor=actor)
        authority.processing_taxonomy_version_id = draft.id
        authority.processing_head_revision = 1
        session.commit()
    return draft.id
