"""
Theme Taxonomy Service — L1/L2 hierarchical theme grouping.

Provides:
- L1 CRUD: create, list, assign/unassign L2→L1
- 3-phase assignment pipeline: rule-based → HDBSCAN clustering → LLM naming
- L1 metrics aggregation via batch SQL
- L1 centroid embeddings (mean of children)
- New L2 → L1 classification via cosine similarity
- L1 retirement policy
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import numpy as np
from sqlalchemy import func, case, distinct
from sqlalchemy.orm import Session

from ..models.theme import (
    ThemeCluster,
    ThemeConstituent,
    ThemeEmbedding,
    ThemeMention,
    ThemeMetrics,
)
from .theme_embedding_service import ThemeEmbeddingEngine, ThemeEmbeddingRepository
from .theme_identity_normalization import canonical_theme_key
from .theme_taxonomy_prompts import build_l1_naming_prompt, L1_CATEGORIES

logger = logging.getLogger(__name__)

# ── Rule-based prefix patterns for Phase 1 assignment ─────────────────
# Maps canonical_key prefixes to L1 display names and categories.
_L1_PREFIX_RULES: list[tuple[str, str, str]] = [
    # (prefix_pattern, L1 display_name, category)
    # AI & Technology
    ("ai_", "AI & Machine Learning", "technology"),
    ("artificial_intelligence", "AI & Machine Learning", "technology"),
    ("machine_learning", "AI & Machine Learning", "technology"),
    ("deep_learning", "AI & Machine Learning", "technology"),
    ("llm", "AI & Machine Learning", "technology"),
    ("large_language_model", "AI & Machine Learning", "technology"),
    ("generative_ai", "AI & Machine Learning", "technology"),
    ("gpu", "AI & Machine Learning", "technology"),
    # Semiconductors
    ("semiconductor", "Semiconductors", "technology"),
    ("chip", "Semiconductors", "technology"),
    ("foundry", "Semiconductors", "technology"),
    ("hbm", "Semiconductors", "technology"),
    # Cloud & Data Centers
    ("cloud", "Cloud & Data Centers", "technology"),
    ("datacenter", "Cloud & Data Centers", "technology"),
    ("data_center", "Cloud & Data Centers", "technology"),
    ("hyperscaler", "Cloud & Data Centers", "technology"),
    # Cybersecurity
    ("cybersecurity", "Cybersecurity", "technology"),
    ("cyber_security", "Cybersecurity", "technology"),
    ("zero_trust", "Cybersecurity", "technology"),
    # Quantum
    ("quantum", "Quantum Computing", "technology"),
    # Software
    ("saas", "Software & SaaS", "technology"),
    ("software", "Software & SaaS", "technology"),
    # Nuclear
    ("nuclear", "Nuclear Energy", "energy"),
    ("uranium", "Nuclear Energy", "energy"),
    ("smr", "Nuclear Energy", "energy"),
    # Solar / Renewables
    ("solar", "Renewable Energy", "energy"),
    ("wind_energy", "Renewable Energy", "energy"),
    ("clean_energy", "Renewable Energy", "energy"),
    ("renewable", "Renewable Energy", "energy"),
    ("green_energy", "Renewable Energy", "energy"),
    # Oil & Gas
    ("oil", "Oil & Gas", "energy"),
    ("natural_gas", "Oil & Gas", "energy"),
    ("lng", "Oil & Gas", "energy"),
    ("petroleum", "Oil & Gas", "energy"),
    # EV
    ("ev_", "Electric Vehicles", "consumer"),
    ("electric_vehicle", "Electric Vehicles", "consumer"),
    ("ev_charging", "Electric Vehicles", "consumer"),
    ("battery", "Electric Vehicles", "consumer"),
    # Defense
    ("defense", "Defense & Aerospace", "defense"),
    ("military", "Defense & Aerospace", "defense"),
    ("drone", "Defense & Aerospace", "defense"),
    ("aerospace", "Defense & Aerospace", "defense"),
    ("munition", "Defense & Aerospace", "defense"),
    # Healthcare
    ("glp1", "GLP-1 & Weight Loss", "healthcare"),
    ("glp_1", "GLP-1 & Weight Loss", "healthcare"),
    ("weight_loss", "GLP-1 & Weight Loss", "healthcare"),
    ("obesity", "GLP-1 & Weight Loss", "healthcare"),
    ("biotech", "Biotech & Pharma", "healthcare"),
    ("pharmaceutical", "Biotech & Pharma", "healthcare"),
    ("gene_therapy", "Biotech & Pharma", "healthcare"),
    # Crypto
    ("bitcoin", "Crypto & Blockchain", "crypto"),
    ("crypto", "Crypto & Blockchain", "crypto"),
    ("blockchain", "Crypto & Blockchain", "crypto"),
    # Financials
    ("fintech", "Fintech", "financials"),
    ("banking", "Banking & Insurance", "financials"),
    ("insurance", "Banking & Insurance", "financials"),
    # Reshoring
    ("reshoring", "Reshoring & Nearshoring", "industrials"),
    ("nearshoring", "Reshoring & Nearshoring", "industrials"),
    ("onshoring", "Reshoring & Nearshoring", "industrials"),
    # Infrastructure
    ("infrastructure", "Infrastructure", "industrials"),
    # Real Estate
    ("real_estate", "Real Estate", "real_estate"),
    ("reit", "Real Estate", "real_estate"),
    ("housing", "Real Estate", "real_estate"),
    # Macro
    ("interest_rate", "Macro & Rates", "macro"),
    ("federal_reserve", "Macro & Rates", "macro"),
    ("inflation", "Macro & Rates", "macro"),
    ("recession", "Macro & Rates", "macro"),
    ("tariff", "Trade & Tariffs", "macro"),
    ("trade_war", "Trade & Tariffs", "macro"),
]

# Minimum thresholds for noise → standalone L1
_NOISE_STANDALONE_MIN_MENTIONS = 3
_NOISE_STANDALONE_MIN_STOCKS = 3
_UNCATEGORIZED_L1_NAME = "Uncategorized Themes"
_UNCATEGORIZED_CATEGORY = "other"

# Cosine similarity threshold for auto-classifying new L2 to L1
_L1_CLASSIFY_THRESHOLD = 0.65

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CENTROID_EMBEDDING_MODEL = f"centroid_{EMBEDDING_MODEL}"


class ThemeTaxonomyService:
    """L1/L2 theme taxonomy management."""

    def __init__(self, db: Session, *, pipeline: str = "technical"):
        self.db = db
        self.pipeline = pipeline
        self.embedding_engine = ThemeEmbeddingEngine(EMBEDDING_MODEL)
        self.embedding_repo = ThemeEmbeddingRepository(db)
        self._l1_centroid_cache: dict[int, np.ndarray] | None = None

    # ── L1 CRUD ───────────────────────────────────────────────────────

    def create_l1_theme(
        self,
        display_name: str,
        category: str,
        description: str = "",
    ) -> ThemeCluster:
        """Create an L1 parent theme. Skips candidate lifecycle — created directly as active."""
        ckey = canonical_theme_key(display_name)
        now = datetime.utcnow()

        existing = self.db.query(ThemeCluster).filter(
            ThemeCluster.pipeline == self.pipeline,
            ThemeCluster.canonical_key == ckey,
            ThemeCluster.is_l1 == True,
        ).first()
        if existing:
            return existing

        theme = ThemeCluster(
            name=display_name,
            canonical_key=ckey,
            display_name=display_name,
            description=description,
            pipeline=self.pipeline,
            category=category,
            is_active=True,
            is_l1=True,
            taxonomy_level=1,
            lifecycle_state="active",
            activated_at=now,
            first_seen_at=now,
            last_seen_at=now,
            discovery_source="taxonomy_assignment",
        )
        self.db.add(theme)
        self.db.flush()
        logger.info("Created L1 theme: %s (id=%d, category=%s)", display_name, theme.id, category)
        return theme

    def get_l1_themes(
        self,
        *,
        category_filter: str | None = None,
        include_children_count: bool = True,
        limit: int = 300,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Return L1 themes with optional children count and aggregated metrics."""
        base_query = self.db.query(ThemeCluster).filter(
            ThemeCluster.pipeline == self.pipeline,
            ThemeCluster.is_l1 == True,
            ThemeCluster.is_active == True,
        )
        if category_filter:
            base_query = base_query.filter(ThemeCluster.category == category_filter)

        total = base_query.count()

        l1_themes = base_query.order_by(
            ThemeCluster.display_name.asc()
        ).offset(offset).limit(limit).all()

        results = []
        for l1 in l1_themes:
            row: dict[str, Any] = {
                "id": l1.id,
                "display_name": l1.display_name,
                "canonical_key": l1.canonical_key,
                "category": l1.category,
                "description": l1.description,
                "activated_at": l1.activated_at.isoformat() if l1.activated_at else None,
                "lifecycle_state": l1.lifecycle_state,
            }

            if include_children_count:
                child_count = self.db.query(func.count(ThemeCluster.id)).filter(
                    ThemeCluster.parent_cluster_id == l1.id,
                    ThemeCluster.is_active == True,
                ).scalar() or 0
                row["num_l2_children"] = child_count

            # Get latest aggregated metrics
            latest_metrics = self.db.query(ThemeMetrics).filter(
                ThemeMetrics.theme_cluster_id == l1.id,
            ).order_by(ThemeMetrics.date.desc()).first()
            if latest_metrics:
                row["mentions_7d"] = latest_metrics.mentions_7d
                row["mentions_30d"] = latest_metrics.mentions_30d
                row["num_constituents"] = latest_metrics.num_constituents
                row["momentum_score"] = latest_metrics.momentum_score
                row["basket_return_1w"] = latest_metrics.basket_return_1w
                row["basket_rs_vs_spy"] = latest_metrics.basket_rs_vs_spy
                row["rank"] = latest_metrics.rank
            else:
                row["mentions_7d"] = 0
                row["mentions_30d"] = 0
                row["num_constituents"] = 0
                row["momentum_score"] = None
                row["basket_return_1w"] = None
                row["basket_rs_vs_spy"] = None
                row["rank"] = None

            results.append(row)

        return results, total

    def get_l1_with_children(
        self,
        l1_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> dict | None:
        """Get an L1 theme with its L2 children (paginated)."""
        l1 = self.db.query(ThemeCluster).filter(
            ThemeCluster.id == l1_id,
            ThemeCluster.is_l1 == True,
        ).first()
        if not l1:
            return None

        children_query = self.db.query(ThemeCluster).filter(
            ThemeCluster.parent_cluster_id == l1_id,
            ThemeCluster.is_active == True,
        )
        total_children = children_query.count()
        children = children_query.order_by(
            ThemeCluster.display_name.asc()
        ).offset(offset).limit(limit).all()

        child_rows = []
        for child in children:
            latest_metrics = self.db.query(ThemeMetrics).filter(
                ThemeMetrics.theme_cluster_id == child.id,
            ).order_by(ThemeMetrics.date.desc()).first()

            child_rows.append({
                "id": child.id,
                "display_name": child.display_name,
                "canonical_key": child.canonical_key,
                "category": child.category,
                "lifecycle_state": child.lifecycle_state,
                "l1_assignment_method": child.l1_assignment_method,
                "l1_assignment_confidence": child.l1_assignment_confidence,
                "mentions_7d": latest_metrics.mentions_7d if latest_metrics else 0,
                "mentions_30d": latest_metrics.mentions_30d if latest_metrics else 0,
                "num_constituents": latest_metrics.num_constituents if latest_metrics else 0,
                "momentum_score": latest_metrics.momentum_score if latest_metrics else None,
            })

        return {
            "l1": {
                "id": l1.id,
                "display_name": l1.display_name,
                "canonical_key": l1.canonical_key,
                "category": l1.category,
                "description": l1.description,
            },
            "children": child_rows,
            "total_children": total_children,
        }

    def assign_l2_to_l1(
        self,
        l2_id: int,
        l1_id: int,
        *,
        method: str = "manual",
        confidence: float = 1.0,
    ) -> bool:
        """Assign an L2 theme to an L1 parent."""
        l2 = self.db.query(ThemeCluster).filter(
            ThemeCluster.id == l2_id,
            ThemeCluster.is_l1 == False,
        ).first()
        if not l2:
            return False

        l1 = self.db.query(ThemeCluster).filter(
            ThemeCluster.id == l1_id,
            ThemeCluster.is_l1 == True,
        ).first()
        if not l1:
            return False

        l2.parent_cluster_id = l1_id
        l2.l1_assignment_method = method
        l2.l1_assignment_confidence = confidence
        l2.l1_assigned_at = datetime.utcnow()
        self.db.flush()

        # Invalidate centroid cache
        self._l1_centroid_cache = None
        return True

    def unassign_l2_from_l1(self, l2_id: int) -> bool:
        """Remove L1 assignment from an L2 theme."""
        l2 = self.db.query(ThemeCluster).filter(
            ThemeCluster.id == l2_id,
            ThemeCluster.is_l1 == False,
        ).first()
        if not l2 or l2.parent_cluster_id is None:
            return False

        l2.parent_cluster_id = None
        l2.l1_assignment_method = None
        l2.l1_assignment_confidence = None
        l2.l1_assigned_at = None
        self.db.flush()

        self._l1_centroid_cache = None
        return True

    def get_unassigned_themes(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Return L2 themes that have no L1 parent."""
        base_query = self.db.query(ThemeCluster).filter(
            ThemeCluster.pipeline == self.pipeline,
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
            ThemeCluster.parent_cluster_id.is_(None),
        )
        total = base_query.count()
        themes = base_query.order_by(
            ThemeCluster.display_name.asc()
        ).offset(offset).limit(limit).all()

        return [
            {
                "id": t.id,
                "display_name": t.display_name,
                "canonical_key": t.canonical_key,
                "category": t.category,
                "lifecycle_state": t.lifecycle_state,
            }
            for t in themes
        ], total

    def get_categories(self) -> list[dict]:
        """Return available L1 categories with counts."""
        rows = self.db.query(
            ThemeCluster.category,
            func.count(ThemeCluster.id),
        ).filter(
            ThemeCluster.pipeline == self.pipeline,
            ThemeCluster.is_l1 == True,
            ThemeCluster.is_active == True,
        ).group_by(ThemeCluster.category).all()

        return [
            {"category": cat or "other", "count": count}
            for cat, count in sorted(rows, key=lambda r: r[0] or "zzz")
        ]

    # ── Full Assignment Pipeline ──────────────────────────────────────

    def run_full_taxonomy_assignment(self, *, dry_run: bool = False) -> dict:
        """
        3-phase L1 taxonomy assignment: rules → HDBSCAN → LLM naming.

        Returns a report dict with proposed/applied assignments.
        """
        # Get all unassigned active L2 themes
        unassigned = self.db.query(ThemeCluster).filter(
            ThemeCluster.pipeline == self.pipeline,
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
            ThemeCluster.parent_cluster_id.is_(None),
        ).all()

        report: dict[str, Any] = {
            "pipeline": self.pipeline,
            "total_unassigned": len(unassigned),
            "dry_run": dry_run,
            "phase1_rule_based": {},
            "phase2_clustering": {},
            "phase3_llm_naming": {},
            "l1_themes_created": 0,
            "l2_themes_assigned": 0,
            "still_unassigned": 0,
        }

        if not unassigned:
            return report

        # Phase 1: Rule-based prefix matching
        rule_assigned, remaining = self._phase1_rule_based(unassigned, dry_run=dry_run)
        report["phase1_rule_based"] = {
            "assigned": len(rule_assigned),
            "assignments": [
                {"l2_id": l2.id, "l2_name": l2.display_name, "l1_name": l1_name}
                for l2, l1_name in rule_assigned
            ],
        }

        # Phase 2: HDBSCAN clustering on remaining unassigned
        cluster_groups, noise_themes = self._phase2_embedding_clustering(remaining)
        report["phase2_clustering"] = {
            "num_clusters": len(cluster_groups),
            "noise_themes": len(noise_themes),
            "clusters": [
                {
                    "members": [t.display_name for t in members],
                    "size": len(members),
                }
                for members in cluster_groups
            ],
        }

        # Phase 3: Name each cluster via LLM (or fallback)
        l1_assignments = self._phase3_llm_naming(cluster_groups, dry_run=dry_run)
        report["phase3_llm_naming"] = {
            "l1_themes": [
                {
                    "l1_name": info["l1_name"],
                    "category": info["category"],
                    "members": [t.display_name for t in info["members"]],
                }
                for info in l1_assignments
            ],
        }

        # Handle noise themes
        if not dry_run:
            self._handle_noise_themes(noise_themes)

        # Count totals
        total_assigned = len(rule_assigned) + sum(len(a["members"]) for a in l1_assignments)
        report["l2_themes_assigned"] = total_assigned
        report["still_unassigned"] = len(unassigned) - total_assigned

        # Count L1 themes created
        if not dry_run:
            l1_count = self.db.query(func.count(ThemeCluster.id)).filter(
                ThemeCluster.pipeline == self.pipeline,
                ThemeCluster.is_l1 == True,
            ).scalar() or 0
            report["l1_themes_created"] = l1_count

        if not dry_run:
            self.db.commit()

        return report

    def _phase1_rule_based(
        self,
        themes: list[ThemeCluster],
        *,
        dry_run: bool = False,
    ) -> tuple[list[tuple[ThemeCluster, str]], list[ThemeCluster]]:
        """
        Phase 1: Assign L2 themes to L1 by canonical_key prefix matching.

        Returns (assigned_pairs, remaining_unassigned).
        """
        assigned: list[tuple[ThemeCluster, str]] = []
        remaining: list[ThemeCluster] = []
        l1_cache: dict[str, ThemeCluster] = {}

        for theme in themes:
            key = (theme.canonical_key or "").strip().lower()
            matched = False

            for prefix, l1_name, category in _L1_PREFIX_RULES:
                if key.startswith(prefix) or key == prefix.rstrip("_"):
                    if not dry_run:
                        if l1_name not in l1_cache:
                            l1_cache[l1_name] = self.create_l1_theme(l1_name, category)
                        l1 = l1_cache[l1_name]
                        self.assign_l2_to_l1(
                            theme.id, l1.id,
                            method="rule_based", confidence=0.9,
                        )
                    assigned.append((theme, l1_name))
                    matched = True
                    break

            if not matched:
                remaining.append(theme)

        return assigned, remaining

    def _phase2_embedding_clustering(
        self,
        themes: list[ThemeCluster],
    ) -> tuple[list[list[ThemeCluster]], list[ThemeCluster]]:
        """
        Phase 2: Cluster remaining unassigned themes using HDBSCAN on embeddings.

        Returns (cluster_groups, noise_themes).
        """
        if len(themes) < 3:
            return [], themes

        # Load embeddings for these themes
        theme_ids = [t.id for t in themes]
        embedding_map = self.embedding_repo.get_by_cluster_ids(theme_ids)

        themes_with_embeddings = []
        vectors = []
        for theme in themes:
            emb = embedding_map.get(theme.id)
            if emb and emb.embedding:
                vec = ThemeEmbeddingEngine.deserialize(emb.embedding)
                if vec is not None:
                    themes_with_embeddings.append(theme)
                    vectors.append(vec)

        if len(themes_with_embeddings) < 3:
            return [], themes

        # HDBSCAN clustering
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            logger.warning("sklearn.cluster.HDBSCAN not available, skipping clustering phase")
            return [], themes

        X = np.array(vectors)
        clusterer = HDBSCAN(
            min_cluster_size=3,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(X)

        # Group by cluster label
        cluster_groups: dict[int, list[ThemeCluster]] = defaultdict(list)
        noise_themes: list[ThemeCluster] = []

        for theme, label in zip(themes_with_embeddings, labels):
            if label == -1:
                noise_themes.append(theme)
            else:
                cluster_groups[label].append(theme)

        # Also include themes without embeddings as noise
        embedded_ids = {t.id for t in themes_with_embeddings}
        for theme in themes:
            if theme.id not in embedded_ids:
                noise_themes.append(theme)

        return list(cluster_groups.values()), noise_themes

    def _phase3_llm_naming(
        self,
        cluster_groups: list[list[ThemeCluster]],
        *,
        dry_run: bool = False,
    ) -> list[dict]:
        """
        Phase 3: Generate L1 names for each cluster using LLM (with non-LLM fallback).

        Returns list of dicts with l1_name, category, members.
        """
        assignments: list[dict] = []

        for group in cluster_groups:
            if not group:
                continue

            # Try LLM naming
            l1_info = self._llm_name_cluster(group)
            if l1_info is None:
                # Fallback: use highest-mention child's display_name
                l1_info = self._fallback_name_cluster(group)

            if not dry_run:
                l1 = self.create_l1_theme(
                    l1_info["l1_name"],
                    l1_info["category"],
                    l1_info.get("description", ""),
                )
                for theme in group:
                    self.assign_l2_to_l1(
                        theme.id, l1.id,
                        method="clustering", confidence=0.7,
                    )

            assignments.append({
                "l1_name": l1_info["l1_name"],
                "category": l1_info["category"],
                "members": group,
            })

        return assignments

    def _llm_name_cluster(self, themes: list[ThemeCluster]) -> dict | None:
        """Use LLM to generate an L1 name and category for a cluster of themes."""
        try:
            from .llm import LLMService, LLMError

            llm = LLMService(use_case="extraction")
            theme_names = [t.display_name for t in themes]
            prompt = build_l1_naming_prompt(theme_names)

            response = llm.completion_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = LLMService.extract_content(response)
            if not content:
                return None

            # Parse JSON response
            text = content.strip()
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)

            parsed = json.loads(text)
            l1_name = parsed.get("l1_name", "").strip()
            category = parsed.get("category", "other").strip().lower()
            description = parsed.get("description", "").strip()

            if not l1_name:
                return None

            if category not in L1_CATEGORIES:
                category = "other"

            return {"l1_name": l1_name, "category": category, "description": description}

        except Exception as e:
            logger.warning("LLM naming failed for cluster, using fallback: %s", e)
            return None

    def _fallback_name_cluster(self, themes: list[ThemeCluster]) -> dict:
        """Fallback naming: use highest-mention child's display_name."""
        # Find child with most mentions
        best = themes[0]
        best_mentions = 0

        for theme in themes:
            mention_count = self.db.query(func.count(ThemeMention.id)).filter(
                ThemeMention.theme_cluster_id == theme.id,
            ).scalar() or 0
            if mention_count > best_mentions:
                best_mentions = mention_count
                best = theme

        return {
            "l1_name": best.display_name,
            "category": best.category or "other",
            "description": f"Group of {len(themes)} related themes",
        }

    def _handle_noise_themes(self, noise_themes: list[ThemeCluster]) -> None:
        """Handle HDBSCAN noise: substantive → standalone L1, rest → Uncategorized."""
        uncategorized_l1: ThemeCluster | None = None

        for theme in noise_themes:
            mention_count = self.db.query(func.count(ThemeMention.id)).filter(
                ThemeMention.theme_cluster_id == theme.id,
            ).scalar() or 0

            stock_count = self.db.query(func.count(ThemeConstituent.id)).filter(
                ThemeConstituent.theme_cluster_id == theme.id,
                ThemeConstituent.is_active == True,
            ).scalar() or 0

            if mention_count >= _NOISE_STANDALONE_MIN_MENTIONS or stock_count >= _NOISE_STANDALONE_MIN_STOCKS:
                # Substantive theme → create standalone L1
                l1 = self.create_l1_theme(
                    theme.display_name,
                    theme.category or "other",
                    f"Standalone theme (mentions={mention_count}, stocks={stock_count})",
                )
                self.assign_l2_to_l1(
                    theme.id, l1.id,
                    method="clustering", confidence=0.5,
                )
            else:
                # Low-signal → Uncategorized bucket
                if uncategorized_l1 is None:
                    uncategorized_l1 = self.create_l1_theme(
                        _UNCATEGORIZED_L1_NAME,
                        _UNCATEGORIZED_CATEGORY,
                        "Collection of low-signal themes awaiting classification",
                    )
                self.assign_l2_to_l1(
                    theme.id, uncategorized_l1.id,
                    method="clustering", confidence=0.3,
                )

    # ── L1 Metrics Aggregation ────────────────────────────────────────

    def compute_all_l1_metrics(self, *, as_of_date: datetime | None = None) -> dict:
        """
        Aggregate L2 metrics → L1 metrics via batch SQL.

        Single pass for all L1 themes using GROUP BY parent_cluster_id.
        """
        now = as_of_date or datetime.utcnow()
        today = now.date()

        # Get all active L1 themes in this pipeline
        l1_themes = self.db.query(ThemeCluster).filter(
            ThemeCluster.pipeline == self.pipeline,
            ThemeCluster.is_l1 == True,
            ThemeCluster.is_active == True,
        ).all()

        if not l1_themes:
            return {"l1_count": 0, "metrics_updated": 0}

        l1_ids = [t.id for t in l1_themes]

        # Find latest metrics date for L2 themes in this pipeline
        latest_l2_date = self.db.query(func.max(ThemeMetrics.date)).filter(
            ThemeMetrics.pipeline == self.pipeline,
        ).scalar()
        if not latest_l2_date:
            return {"l1_count": len(l1_ids), "metrics_updated": 0}

        # Batch aggregation: GROUP BY parent_cluster_id
        # Join L2 clusters → their latest metrics
        agg_rows = self.db.query(
            ThemeCluster.parent_cluster_id,
            func.sum(ThemeMetrics.mentions_7d).label("sum_mentions_7d"),
            func.sum(ThemeMetrics.mentions_30d).label("sum_mentions_30d"),
            func.sum(ThemeMetrics.mentions_1d).label("sum_mentions_1d"),
            func.avg(ThemeMetrics.mention_velocity).label("avg_velocity"),
            func.avg(ThemeMetrics.sentiment_score).label("avg_sentiment"),
            func.avg(ThemeMetrics.basket_return_1d).label("avg_return_1d"),
            func.avg(ThemeMetrics.basket_return_1w).label("avg_return_1w"),
            func.avg(ThemeMetrics.basket_return_1m).label("avg_return_1m"),
            func.avg(ThemeMetrics.basket_rs_vs_spy).label("avg_rs"),
            func.avg(ThemeMetrics.avg_internal_correlation).label("avg_corr"),
            func.sum(ThemeMetrics.num_constituents).label("sum_constituents"),
            func.avg(ThemeMetrics.pct_above_50ma).label("avg_pct_50ma"),
            func.avg(ThemeMetrics.pct_above_200ma).label("avg_pct_200ma"),
            func.avg(ThemeMetrics.pct_positive_1w).label("avg_pct_pos_1w"),
            func.sum(ThemeMetrics.num_passing_minervini).label("sum_minervini"),
            func.sum(ThemeMetrics.num_stage_2).label("sum_stage2"),
            func.avg(ThemeMetrics.avg_rs_rating).label("avg_rs_rating"),
            func.avg(ThemeMetrics.momentum_score).label("avg_momentum"),
            func.count(ThemeMetrics.id).label("child_count"),
        ).join(
            ThemeMetrics, ThemeMetrics.theme_cluster_id == ThemeCluster.id,
        ).filter(
            ThemeCluster.parent_cluster_id.in_(l1_ids),
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
            ThemeMetrics.date == latest_l2_date,
        ).group_by(
            ThemeCluster.parent_cluster_id,
        ).all()

        # Also get distinct constituent counts per L1
        constituent_counts = self.db.query(
            ThemeCluster.parent_cluster_id,
            func.count(distinct(ThemeConstituent.symbol)).label("unique_symbols"),
        ).join(
            ThemeConstituent, ThemeConstituent.theme_cluster_id == ThemeCluster.id,
        ).filter(
            ThemeCluster.parent_cluster_id.in_(l1_ids),
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
            ThemeConstituent.is_active == True,
        ).group_by(
            ThemeCluster.parent_cluster_id,
        ).all()
        constituent_map = {row[0]: row[1] for row in constituent_counts}

        # Build lookup from aggregation
        agg_map = {row.parent_cluster_id: row for row in agg_rows}

        metrics_updated = 0
        for l1 in l1_themes:
            agg = agg_map.get(l1.id)
            if not agg:
                continue

            unique_constituents = constituent_map.get(l1.id, 0)

            # Upsert L1 metrics
            existing = self.db.query(ThemeMetrics).filter(
                ThemeMetrics.theme_cluster_id == l1.id,
                ThemeMetrics.date == today,
            ).first()

            def _safe_float(val):
                return round(float(val), 4) if val is not None else None

            def _safe_int(val):
                return int(val) if val is not None else 0

            metrics_data = {
                "pipeline": self.pipeline,
                "mentions_1d": _safe_int(agg.sum_mentions_1d),
                "mentions_7d": _safe_int(agg.sum_mentions_7d),
                "mentions_30d": _safe_int(agg.sum_mentions_30d),
                "mention_velocity": _safe_float(agg.avg_velocity),
                "sentiment_score": _safe_float(agg.avg_sentiment),
                "basket_return_1d": _safe_float(agg.avg_return_1d),
                "basket_return_1w": _safe_float(agg.avg_return_1w),
                "basket_return_1m": _safe_float(agg.avg_return_1m),
                "basket_rs_vs_spy": _safe_float(agg.avg_rs),
                "avg_internal_correlation": _safe_float(agg.avg_corr),
                "num_constituents": unique_constituents,
                "pct_above_50ma": _safe_float(agg.avg_pct_50ma),
                "pct_above_200ma": _safe_float(agg.avg_pct_200ma),
                "pct_positive_1w": _safe_float(agg.avg_pct_pos_1w),
                "num_passing_minervini": _safe_int(agg.sum_minervini),
                "num_stage_2": _safe_int(agg.sum_stage2),
                "avg_rs_rating": _safe_float(agg.avg_rs_rating),
                "momentum_score": _safe_float(agg.avg_momentum),
            }

            if existing:
                for key, value in metrics_data.items():
                    setattr(existing, key, value)
            else:
                metrics = ThemeMetrics(
                    theme_cluster_id=l1.id,
                    date=today,
                    **metrics_data,
                )
                self.db.add(metrics)

            metrics_updated += 1

        self.db.flush()
        return {"l1_count": len(l1_ids), "metrics_updated": metrics_updated}

    # ── L1 Centroid Embeddings ────────────────────────────────────────

    def compute_l1_centroid_embeddings(self) -> dict:
        """
        Compute L1 centroid embeddings as mean of children's embeddings.

        Stored in ThemeEmbedding with embedding_model="centroid_all-MiniLM-L6-v2".
        """
        l1_themes = self.db.query(ThemeCluster).filter(
            ThemeCluster.pipeline == self.pipeline,
            ThemeCluster.is_l1 == True,
            ThemeCluster.is_active == True,
        ).all()

        updated = 0
        skipped = 0

        for l1 in l1_themes:
            children = self.db.query(ThemeCluster).filter(
                ThemeCluster.parent_cluster_id == l1.id,
                ThemeCluster.is_active == True,
            ).all()

            if not children:
                skipped += 1
                continue

            child_ids = [c.id for c in children]
            embedding_map = self.embedding_repo.get_by_cluster_ids(child_ids)

            vectors = []
            for child_id in child_ids:
                emb = embedding_map.get(child_id)
                if emb and emb.embedding:
                    vec = ThemeEmbeddingEngine.deserialize(emb.embedding)
                    if vec is not None:
                        vectors.append(vec)

            if not vectors:
                skipped += 1
                continue

            centroid = np.mean(vectors, axis=0)
            centroid_json = ThemeEmbeddingEngine.serialize(centroid)
            child_names = ", ".join(c.display_name for c in children[:5])
            embedding_text = f"L1 centroid of: {child_names}"

            # Upsert centroid embedding for the L1 theme
            self.embedding_repo.upsert_for_theme(
                l1,
                embedding_json=centroid_json,
                embedding_text=embedding_text,
                embedding_model=CENTROID_EMBEDDING_MODEL,
                content_hash=f"centroid_{l1.id}_{len(vectors)}",
                model_version="centroid-v1",
                is_stale=False,
            )
            updated += 1

        self.db.flush()
        self._l1_centroid_cache = None  # Invalidate
        return {"l1_count": len(l1_themes), "updated": updated, "skipped": skipped}

    def _load_l1_centroid_cache(self) -> dict[int, np.ndarray]:
        """Load all L1 centroid embeddings into memory cache."""
        if self._l1_centroid_cache is not None:
            return self._l1_centroid_cache

        l1_ids = [
            row[0] for row in
            self.db.query(ThemeCluster.id).filter(
                ThemeCluster.pipeline == self.pipeline,
                ThemeCluster.is_l1 == True,
                ThemeCluster.is_active == True,
            ).all()
        ]

        if not l1_ids:
            self._l1_centroid_cache = {}
            return self._l1_centroid_cache

        embeddings = self.db.query(ThemeEmbedding).filter(
            ThemeEmbedding.theme_cluster_id.in_(l1_ids),
            ThemeEmbedding.embedding_model == CENTROID_EMBEDDING_MODEL,
        ).all()

        cache: dict[int, np.ndarray] = {}
        for emb in embeddings:
            vec = ThemeEmbeddingEngine.deserialize(emb.embedding)
            if vec is not None:
                cache[emb.theme_cluster_id] = vec

        self._l1_centroid_cache = cache
        return cache

    # ── New L2 → L1 Classification ────────────────────────────────────

    def classify_new_l2_to_l1(self, l2_cluster: ThemeCluster) -> int | None:
        """
        Classify a new L2 theme to its best-matching L1 parent via cosine similarity.

        Returns L1 id if assigned, None if no match above threshold.
        """
        if l2_cluster.is_l1 or l2_cluster.parent_cluster_id is not None:
            return l2_cluster.parent_cluster_id

        # Get L2 embedding
        l2_emb = self.embedding_repo.get_for_cluster(l2_cluster.id)
        if not l2_emb or not l2_emb.embedding:
            return None

        l2_vec = ThemeEmbeddingEngine.deserialize(l2_emb.embedding)
        if l2_vec is None:
            return None

        # Load L1 centroids
        centroids = self._load_l1_centroid_cache()
        if not centroids:
            return None

        # Find best match
        best_l1_id: int | None = None
        best_score: float = 0.0

        for l1_id, centroid_vec in centroids.items():
            score = ThemeEmbeddingEngine.cosine_similarity(l2_vec, centroid_vec)
            if score > best_score:
                best_score = score
                best_l1_id = l1_id

        if best_l1_id is not None and best_score >= _L1_CLASSIFY_THRESHOLD:
            self.assign_l2_to_l1(
                l2_cluster.id, best_l1_id,
                method="clustering", confidence=round(best_score, 4),
            )
            logger.info(
                "Auto-classified L2 '%s' → L1 id=%d (score=%.3f)",
                l2_cluster.display_name, best_l1_id, best_score,
            )
            return best_l1_id

        return None

    # ── L1 Retirement ─────────────────────────────────────────────────

    def check_l1_retirement(self, l1_id: int) -> bool:
        """
        Retire L1 only when ALL active L2 children are retired.

        L1 themes never go dormant — only active or retired.
        """
        l1 = self.db.query(ThemeCluster).filter(
            ThemeCluster.id == l1_id,
            ThemeCluster.is_l1 == True,
        ).first()
        if not l1:
            return False

        active_children = self.db.query(func.count(ThemeCluster.id)).filter(
            ThemeCluster.parent_cluster_id == l1_id,
            ThemeCluster.is_active == True,
            ThemeCluster.lifecycle_state != "retired",
        ).scalar() or 0

        if active_children == 0:
            # All children retired → retire L1
            total_children = self.db.query(func.count(ThemeCluster.id)).filter(
                ThemeCluster.parent_cluster_id == l1_id,
            ).scalar() or 0
            if total_children > 0:
                l1.lifecycle_state = "retired"
                l1.retired_at = datetime.utcnow()
                l1.is_active = False
                self.db.flush()
                logger.info("Retired L1 theme: %s (id=%d)", l1.display_name, l1.id)
                return True

        return False
