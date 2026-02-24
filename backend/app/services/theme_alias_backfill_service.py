"""Backfill service for canonical alias rows from historical clusters and mentions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ..models.theme import ThemeAlias, ThemeCluster, ThemeMention
from .theme_identity_normalization import UNKNOWN_THEME_KEY, canonical_theme_key


@dataclass
class _CandidateAggregate:
    pipeline: str
    alias_key: str
    theme_cluster_id: int
    source_counts: Counter[str] = field(default_factory=Counter)
    text_counts: Counter[str] = field(default_factory=Counter)
    evidence_count: int = 0

    def observe(self, *, alias_text: str, source: str) -> None:
        if alias_text:
            self.text_counts[alias_text] += 1
        self.source_counts[source] += 1
        self.evidence_count += 1

    def representative_text(self) -> str:
        if not self.text_counts:
            return self.alias_key.replace("_", " ").title()
        best = sorted(
            self.text_counts.items(),
            key=lambda item: (-item[1], len(item[0]), item[0].lower()),
        )
        return best[0][0]

    def representative_source(self) -> str:
        if not self.source_counts:
            return "backfill_theme_aliases"
        best = sorted(
            self.source_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return best[0][0]


class ThemeAliasBackfillService:
    """Build and apply reversible alias backfill plans."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def run(self, *, dry_run: bool = False, mention_limit: int = 0) -> dict[str, Any]:
        cluster_map: dict[int, dict[str, str]] = {}
        stats = {
            "clusters_scanned": 0,
            "mentions_scanned": 0,
            "candidate_observations": 0,
            "unknown_key_observations": 0,
            "skipped_missing_cluster_mentions": 0,
            "pipeline_mismatch_mentions": 0,
        }
        existing_by_key: dict[tuple[str, str], dict[str, Any]] = {
            (pipeline, alias_key): {"theme_cluster_id": int(theme_cluster_id)}
            for pipeline, alias_key, theme_cluster_id in (
                self.db.query(ThemeAlias.pipeline, ThemeAlias.alias_key, ThemeAlias.theme_cluster_id)
                .order_by(ThemeAlias.id.asc())
                .yield_per(1000)
            )
        }

        candidates: dict[tuple[str, str, int], _CandidateAggregate] = {}
        collisions: list[dict[str, Any]] = []

        def observe(*, pipeline: str, theme_cluster_id: int, alias_text: str, source: str) -> None:
            key = canonical_theme_key(alias_text)
            if key == UNKNOWN_THEME_KEY:
                stats["unknown_key_observations"] += 1
                return
            aggregate_key = (pipeline, key, theme_cluster_id)
            aggregate = candidates.get(aggregate_key)
            if aggregate is None:
                aggregate = _CandidateAggregate(
                    pipeline=pipeline,
                    alias_key=key,
                    theme_cluster_id=theme_cluster_id,
                )
                candidates[aggregate_key] = aggregate
            aggregate.observe(alias_text=alias_text.strip(), source=source)
            stats["candidate_observations"] += 1

        for cluster_id, pipeline, display_name, name, aliases in (
            self.db.query(
                ThemeCluster.id,
                ThemeCluster.pipeline,
                ThemeCluster.display_name,
                ThemeCluster.name,
                ThemeCluster.aliases,
            )
            .order_by(ThemeCluster.id.asc())
            .yield_per(1000)
        ):
            stats["clusters_scanned"] += 1
            cluster_id = int(cluster_id)
            cluster_pipeline = (pipeline or "technical").strip()
            cluster_map[cluster_id] = {"pipeline": cluster_pipeline}
            observe(
                pipeline=cluster_pipeline,
                theme_cluster_id=cluster_id,
                alias_text=(display_name or name or ""),
                source="cluster_display_name",
            )
            if name:
                observe(
                    pipeline=cluster_pipeline,
                    theme_cluster_id=cluster_id,
                    alias_text=name,
                    source="cluster_name",
                )
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias.strip():
                        observe(
                            pipeline=cluster_pipeline,
                            theme_cluster_id=cluster_id,
                            alias_text=alias,
                            source="cluster_alias",
                        )

        mention_query = self.db.query(
            ThemeMention.id,
            ThemeMention.theme_cluster_id,
            ThemeMention.pipeline,
            ThemeMention.raw_theme,
        ).filter(ThemeMention.theme_cluster_id.isnot(None)).order_by(ThemeMention.id.asc())
        if mention_limit and mention_limit > 0:
            mention_query = mention_query.limit(int(mention_limit))

        for mention_id, mention_cluster_id, mention_pipeline_raw, mention_raw_theme in mention_query.yield_per(1000):
            stats["mentions_scanned"] += 1
            cluster_id = int(mention_cluster_id)
            cluster_meta = cluster_map.get(cluster_id)
            if cluster_meta is None:
                stats["skipped_missing_cluster_mentions"] += 1
                continue

            mention_pipeline = (mention_pipeline_raw or "").strip()
            cluster_pipeline = cluster_meta["pipeline"]
            if mention_pipeline and mention_pipeline != cluster_pipeline:
                stats["pipeline_mismatch_mentions"] += 1
                collisions.append(
                    {
                        "bucket": "pipeline_mismatch",
                        "pipeline": mention_pipeline,
                        "alias_key": canonical_theme_key(mention_raw_theme or ""),
                        "theme_cluster_id": cluster_id,
                        "mention_id": mention_id,
                        "remediation_action": (
                            "Review mention pipeline mismatch and reassign mention or cluster pipeline before alias import."
                        ),
                    }
                )
                continue

            observe(
                pipeline=cluster_pipeline,
                theme_cluster_id=cluster_id,
                alias_text=(mention_raw_theme or ""),
                source="mention_raw_theme",
            )

        by_pipeline_key: dict[tuple[str, str], list[_CandidateAggregate]] = defaultdict(list)
        for aggregate in candidates.values():
            by_pipeline_key[(aggregate.pipeline, aggregate.alias_key)].append(aggregate)

        planned_inserts: list[ThemeAlias] = []
        existing_matches = 0
        for (pipeline, alias_key), aggregates in sorted(by_pipeline_key.items()):
            existing_row = existing_by_key.get((pipeline, alias_key))
            if len(aggregates) > 1:
                collisions.append(
                    {
                        "bucket": "candidate_cluster_collision",
                        "pipeline": pipeline,
                        "alias_key": alias_key,
                        "theme_cluster_ids": sorted(a.theme_cluster_id for a in aggregates),
                        "candidate_samples": [
                            {
                                "theme_cluster_id": a.theme_cluster_id,
                                "alias_text": a.representative_text(),
                                "evidence_count": a.evidence_count,
                            }
                            for a in sorted(aggregates, key=lambda row: row.theme_cluster_id)
                        ],
                        "existing_theme_cluster_id": existing_row["theme_cluster_id"] if existing_row else None,
                        "remediation_action": (
                            "Queue for manual merge review; keep aliases unchanged until a canonical winner is approved."
                        ),
                    }
                )
                continue

            aggregate = aggregates[0]
            if existing_row is not None:
                if int(existing_row["theme_cluster_id"]) == int(aggregate.theme_cluster_id):
                    existing_matches += 1
                    continue
                collisions.append(
                    {
                        "bucket": "existing_alias_conflict",
                        "pipeline": pipeline,
                        "alias_key": alias_key,
                        "existing_theme_cluster_id": int(existing_row["theme_cluster_id"]),
                        "candidate_theme_cluster_id": int(aggregate.theme_cluster_id),
                        "candidate_alias_text": aggregate.representative_text(),
                        "remediation_action": (
                            "Preserve existing alias mapping and place in conflict review bucket for manual resolution."
                        ),
                    }
                )
                continue

            planned_inserts.append(
                ThemeAlias(
                    theme_cluster_id=aggregate.theme_cluster_id,
                    pipeline=pipeline,
                    alias_text=aggregate.representative_text(),
                    alias_key=alias_key,
                    source=aggregate.representative_source(),
                    confidence=0.6,
                    evidence_count=max(1, aggregate.evidence_count),
                    first_seen_at=None,
                    last_seen_at=datetime.utcnow(),
                    is_active=True,
                )
            )

        if not dry_run and planned_inserts:
            self.db.add_all(planned_inserts)
            self.db.commit()
        elif not dry_run:
            # Keep transaction boundaries explicit for callers even when no inserts are required.
            self.db.commit()

        collision_buckets = Counter(item["bucket"] for item in collisions)
        remediation_actions = [
            {
                "bucket": bucket,
                "count": count,
                "action": _bucket_action(bucket),
            }
            for bucket, count in sorted(collision_buckets.items())
        ]

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "dry_run": bool(dry_run),
            "totals": {
                **stats,
                "candidate_groups": len(by_pipeline_key),
                "existing_matches": existing_matches,
                "planned_inserts": len(planned_inserts),
                "inserted": 0 if dry_run else len(planned_inserts),
                "collisions_total": len(collisions),
            },
            "collisions": {
                "by_bucket": dict(sorted(collision_buckets.items())),
                "inventory": collisions,
            },
            "remediation_actions": remediation_actions,
        }


def _bucket_action(bucket: str) -> str:
    actions = {
        "candidate_cluster_collision": (
            "Run merge-review workflow to pick winning cluster, then rerun backfill."
        ),
        "existing_alias_conflict": (
            "Keep current alias mapping and add ticket for manual reassignment if needed."
        ),
        "pipeline_mismatch": (
            "Correct mention/cluster pipeline assignment before creating alias rows."
        ),
    }
    return actions.get(bucket, "Manual review required.")
