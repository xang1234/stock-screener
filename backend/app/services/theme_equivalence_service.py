"""Reviewed equivalence operations; original cluster assignments never move."""

from datetime import datetime, timezone

from app.models.theme import ThemeCluster, ThemeMention
from app.models.theme_intelligence import ThemeEquivalenceOperation


class EquivalenceConflict(ValueError):
    pass


class ThemeEquivalenceService:
    def __init__(self, db):
        self.db = db

    def operations(self, pipeline=None):
        query = self.db.query(ThemeEquivalenceOperation)
        if pipeline:
            query = query.filter(ThemeEquivalenceOperation.pipeline == pipeline)
        return query.order_by(ThemeEquivalenceOperation.id).all()

    def snapshot(self, pipeline=None):
        from .theme_group_snapshot import ThemeGroupSnapshot

        return ThemeGroupSnapshot.read(self.db, pipeline)

    def mapping(self, pipeline=None):
        return dict(self.snapshot(pipeline).mapping)

    def version(self, pipeline=None):
        return self.snapshot(pipeline).version

    def representative(self, theme_id):
        return self.snapshot().representative(theme_id)

    def members(self, theme_id):
        return list(self.snapshot().members(theme_id))

    def _lock(self):
        from .theme_group_coordination import lock_grouping_mutation

        lock_grouping_mutation(self.db)
        # Serialize membership edits with a consistent lock order. Extraction keeps
        # writing original IDs and needs no group lock.
        self.db.query(ThemeCluster).order_by(ThemeCluster.id).with_for_update().all()

    def preview(self, source_id, target_id):
        source, target = (
            self.db.get(ThemeCluster, source_id),
            self.db.get(ThemeCluster, target_id),
        )
        if source is None or target is None:
            raise EquivalenceConflict("Theme not found")
        if source.pipeline != target.pipeline:
            raise EquivalenceConflict("Cross-pipeline grouping is not allowed")
        group = self.snapshot(source.pipeline)
        members = group.expand([source_id, target_id])
        rows = self.db.query(ThemeCluster).filter(ThemeCluster.id.in_(members)).all()
        if any(not row.is_active or row.is_l1 for row in rows):
            raise EquivalenceConflict("Only active non-parent themes can be grouped")
        # Follow the full parent chain, not only immediate HBM -> Memory links.
        all_rows = {row.id: row for row in self.db.query(ThemeCluster).all()}
        for row in rows:
            seen = {row.id}
            parent = row.parent_cluster_id
            while parent:
                if parent in seen or parent in members:
                    raise EquivalenceConflict(
                        "Hierarchy conflict: broader and narrower themes stay separate"
                    )
                seen.add(parent)
                parent = (
                    all_rows[parent].parent_cluster_id if parent in all_rows else None
                )
        target_id = group.representative(target_id)
        parents = (
            self.db.query(ThemeMention.content_item_id)
            .filter(
                ThemeMention.theme_cluster_id.in_(members),
                ThemeMention.pipeline == source.pipeline,
            )
            .distinct()
            .count()
        )
        return {
            "source_id": source_id,
            "target_id": target_id,
            "pipeline": source.pipeline,
            "member_ids": members,
            "parent_posts": parents,
            "version": group.version,
            "aliases": [
                {
                    "theme_id": row.id,
                    "name": row.display_name or row.name,
                    "aliases": row.aliases or [],
                }
                for row in rows
            ],
        }

    @staticmethod
    def _attribution(actor, reason):
        if not actor.strip() or not reason.strip():
            raise EquivalenceConflict("Reviewer and reason are required")

    def apply(self, source_id, target_id, *, actor, reason, key, expected_version=None):
        self._attribution(actor, reason)
        if not key or len(key) > 120:
            raise EquivalenceConflict("Invalid operation key")
        self._lock()
        old = (
            self.db.query(ThemeEquivalenceOperation)
            .filter_by(operation_key=key)
            .first()
        )
        if old:
            if old.source_id != source_id or old.requested_target_id != target_id:
                raise EquivalenceConflict(
                    "Operation key was used for a different grouping"
                )
            return {"id": old.id, "active": old.active, "pipeline": old.pipeline}
        preview = self.preview(source_id, target_id)
        if expected_version is not None and expected_version != preview["version"]:
            raise EquivalenceConflict("Grouping changed; refresh the preview")
        group = self.snapshot(preview["pipeline"])
        if group.representative(source_id) == group.representative(target_id):
            raise EquivalenceConflict("Themes already belong to the same group")
        row = ThemeEquivalenceOperation(
            operation_key=key,
            source_id=source_id,
            target_id=preview["target_id"],
            requested_target_id=target_id,
            pipeline=preview["pipeline"],
            member_ids=preview["member_ids"],
            aliases=preview["aliases"],
            actor=actor.strip(),
            reason=reason.strip(),
            active=True,
        )
        self.db.add(row)
        self.db.flush()
        return {"id": row.id, "active": True, "pipeline": row.pipeline}

    def undo(self, operation_id, *, actor, reason):
        self._attribution(actor, reason)
        self._lock()
        row = self.db.get(
            ThemeEquivalenceOperation, operation_id, populate_existing=True
        )
        if row is None:
            raise EquivalenceConflict("Grouping operation not found")
        if not row.active:
            return {"id": row.id, "active": False, "pipeline": row.pipeline}
        for later in self.operations(row.pipeline):
            if (
                later.active
                and later.id > row.id
                and set(later.member_ids).intersection(row.member_ids)
            ):
                raise EquivalenceConflict(f"Undo later operation {later.id} first")
        row.refresh_pending = True
        row.active = False
        row.undone_at = datetime.now(timezone.utc)
        row.undone_by, row.undo_reason = actor.strip(), reason.strip()
        self.db.flush()
        return {"id": row.id, "active": False, "pipeline": row.pipeline}


def guard_grouped_merge(db, source_id, target_id):
    service = ThemeEquivalenceService(db)
    service._lock()
    mapping = service.mapping()
    if source_id in mapping or target_id in mapping:
        from app.services.errors import ThemeMergeConflictError

        raise ThemeMergeConflictError(
            "Use reversible equivalence actions for grouped themes",
            error_code="theme_merge_grouped_identity",
        )
