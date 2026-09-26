"""Source dependency for cited evidence (spec §5.3; case E13).

The same original report obtained from an exchange, the issuer's site and a
translation is one origin, not three corroborations. Documents share an
origin when any of their revisions have identical bytes, or when a recorded
document relation (translation, exact mirror, correction, supersession,
amendment, unknown duplicate) connects them. The count this produces is a
dependency-aware origin count; it makes no statistical-independence claim.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models.company_exposure import (
    DocumentRelationRevision,
    ExposureDocumentRevision,
    ExposurePassage,
)

MAX_RELATION_DOCUMENTS = 200


class _UnionFind:
    def __init__(self):
        self.parent: dict[UUID, UUID] = {}

    def find(self, node: UUID) -> UUID:
        self.parent.setdefault(node, node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: UUID, right: UUID) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            # Deterministic representative: the smaller UUID string.
            if str(a) < str(b):
                self.parent[b] = a
            else:
                self.parent[a] = b


def origin_groups(session: Session, passage_ids: Iterable[UUID]) -> dict[UUID, str]:
    """Map each passage to a stable origin-group key."""

    passage_ids = list(dict.fromkeys(passage_ids))
    if not passage_ids:
        return {}
    rows = session.execute(
        select(ExposurePassage.id, ExposureDocumentRevision.document_id)
        .join(
            ExposureDocumentRevision,
            ExposureDocumentRevision.id == ExposurePassage.document_revision_id,
        )
        .where(ExposurePassage.id.in_(passage_ids))
    ).all()
    document_of = {passage_id: document_id for passage_id, document_id in rows}
    groups = _UnionFind()
    frontier = set(document_of.values())
    seen: set[UUID] = set()
    while frontier and len(seen) < MAX_RELATION_DOCUMENTS:
        seen |= frontier
        for document_id in frontier:
            groups.find(document_id)
        related = set()
        relations = session.execute(
            select(
                DocumentRelationRevision.from_document_id,
                DocumentRelationRevision.to_document_id,
            ).where(
                or_(
                    DocumentRelationRevision.from_document_id.in_(frontier),
                    DocumentRelationRevision.to_document_id.in_(frontier),
                )
            )
        ).all()
        for left, right in relations:
            groups.union(left, right)
            related |= {left, right}
        hashes = session.execute(
            select(ExposureDocumentRevision.content_hash).where(
                ExposureDocumentRevision.document_id.in_(frontier)
            )
        ).scalars()
        same_bytes = session.execute(
            select(
                ExposureDocumentRevision.document_id,
                ExposureDocumentRevision.content_hash,
            ).where(ExposureDocumentRevision.content_hash.in_(set(hashes)))
        ).all()
        by_hash: dict[str, list[UUID]] = {}
        for document_id, digest in same_bytes:
            by_hash.setdefault(digest, []).append(document_id)
        for documents in by_hash.values():
            for other in documents[1:]:
                groups.union(documents[0], other)
            related |= set(documents)
        frontier = related - seen
    return {
        passage_id: str(groups.find(document_id))
        for passage_id, document_id in document_of.items()
    }


__all__ = ("origin_groups",)
