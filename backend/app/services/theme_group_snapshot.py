"""One immutable grouping projection for a read or calculation."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from sqlalchemy import case

from app.models.theme_intelligence import ThemeEquivalenceOperation


@dataclass(frozen=True)
class ThemeGroupSnapshot:
    mapping: Mapping[int, int]
    groups: Mapping[int, tuple[int, ...]]
    version: str

    @classmethod
    def read(cls, db, pipeline=None):
        operations = (
            db.query(ThemeEquivalenceOperation)
            .populate_existing()
            .order_by(ThemeEquivalenceOperation.id)
            .all()
        )
        relevant = [
            operation
            for operation in operations
            if pipeline is None or operation.pipeline == pipeline
        ]
        mapping = {}
        for operation in relevant:
            if operation.active:
                mapping.update(
                    {member: operation.target_id for member in operation.member_ids}
                )
        groups = {}
        for member, root in mapping.items():
            groups.setdefault(root, set()).update((member, root))
        version = sha256(
            json.dumps([(op.id, op.active) for op in relevant]).encode()
        ).hexdigest()
        return cls(
            MappingProxyType(mapping),
            MappingProxyType(
                {root: tuple(sorted(ids)) for root, ids in groups.items()}
            ),
            version,
        )

    def representative(self, theme_id):
        return self.mapping.get(theme_id, theme_id)

    def members(self, theme_id):
        root = self.representative(theme_id)
        return self.groups.get(root, (root,))

    def expand(self, theme_ids):
        return sorted(
            {member for theme_id in theme_ids for member in self.members(theme_id)}
        )

    def identity(self, column):
        return (
            case(dict(self.mapping), value=column, else_=column)
            if self.mapping
            else column
        )

    def visible(self, column):
        return ~column.in_(
            [member for member, root in self.mapping.items() if member != root]
        )
