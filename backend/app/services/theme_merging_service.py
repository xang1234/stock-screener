"""
Theme Merging Service

Uses semantic embeddings and LLM verification to identify and merge duplicate themes.
This implements a hybrid approach:
1. Generate embeddings with sentence-transformers (local, free)
2. Find similar themes via cosine similarity
3. Verify with LLM (via LLMService) for high-confidence decisions
4. Queue moderate-confidence pairs for human review
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Callable, Optional

# Disable MPS/Metal before importing PyTorch to avoid fork() issues on macOS
# This must be set before any PyTorch imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_
from sqlalchemy.exc import IntegrityError

from ..models.app_settings import AppSetting
from ..models.theme import (
    ThemeCluster,
    ThemeConstituent,
    ThemeMention,
    ThemeEmbedding,
    ThemeMergeSuggestion,
    ThemeMergeHistory,
)
from .llm import LLMService, LLMError
from .theme_embedding_service import ThemeEmbeddingEngine, ThemeEmbeddingRepository
from .theme_identity_normalization import UNKNOWN_THEME_KEY, canonical_theme_key
from .theme_lifecycle_service import apply_lifecycle_transition

logger = logging.getLogger(__name__)


# LLM prompt for merge verification
MERGE_VERIFICATION_PROMPT = """Analyze whether two market themes should be merged into a single theme.

Theme 1:
- Name: {theme1_name}
- Aliases: {theme1_aliases}
- Description: {theme1_description}
- Category: {theme1_category}

Theme 2:
- Name: {theme2_name}
- Aliases: {theme2_aliases}
- Description: {theme2_description}
- Category: {theme2_category}

Embedding Similarity Score: {similarity:.3f}

Determine the relationship between these themes:
- "identical": Same concept with different wording (e.g., "Defense Stocks" vs "Military & Defense") -> SHOULD MERGE
- "subset": One theme is a subset of the other (e.g., "AI Chips" is subset of "AI Infrastructure") -> DON'T MERGE, but note relationship
- "related": Related but distinct investment themes (e.g., "Solar Energy" vs "Clean Energy") -> DON'T MERGE
- "distinct": Different concepts that happen to have similar names -> DON'T MERGE

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "should_merge": boolean,
  "confidence": float (0.0 to 1.0),
  "relationship": "identical" | "subset" | "related" | "distinct",
  "reasoning": "brief explanation (1-2 sentences)",
  "canonical_name": "suggested merged name if should_merge is true, else null"
}}"""


class ThemeMergingService:
    """Service for identifying and merging duplicate themes using embeddings + LLM"""

    # Similarity thresholds
    EMBEDDING_THRESHOLD = 0.85      # Queue for LLM verification
    AUTO_MERGE_THRESHOLD = 0.95     # Very high similarity candidates
    LLM_CONFIDENCE_THRESHOLD = 0.90 # Required for auto-merge

    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_VERSION = "embedding-v1"
    EMBEDDING_DIM = 384
    DEFAULT_MERGE_MODEL = "groq/llama-3.3-70b-versatile"
    CANDIDATE_TOP_K = 16
    CANDIDATE_BLOCK_MAX = 96

    def __init__(self, db: Session):
        self.db = db
        self.embedding_engine = ThemeEmbeddingEngine(self.EMBEDDING_MODEL)
        self.embedding_repo = ThemeEmbeddingRepository(db)
        self.llm = None
        self.merge_model_id = self.DEFAULT_MERGE_MODEL
        self._load_merge_model_config()
        self._init_llm_client()

        # Rate limiting for LLM calls
        self._last_llm_request = 0
        self._min_llm_interval = 0.5  # seconds
        self._last_pair_generation_metrics: dict[str, object] = {}

    @staticmethod
    def _stable_candidate_rank(source_id: int, candidate_id: int) -> int:
        # Deterministic lightweight rank to avoid id-order bias.
        return ((candidate_id * 1103515245) ^ (source_id * 12345)) & 0x7FFFFFFF

    def _cap_candidate_ids(self, candidate_ids: set[int], *, source_id: int, limit: int) -> set[int]:
        if limit <= 0 or len(candidate_ids) <= limit:
            return set(candidate_ids)
        ranked = sorted(candidate_ids, key=lambda candidate_id: self._stable_candidate_rank(source_id, candidate_id))
        return set(ranked[:limit])

    def _fallback_neighbor_ids(self, *, all_ids: list[int], source_id: int, limit: int) -> set[int]:
        if limit <= 0 or len(all_ids) <= 1:
            return set()
        try:
            idx = all_ids.index(source_id)
        except ValueError:
            return set()
        picked: set[int] = set()
        step = 1
        while len(picked) < limit and (idx - step >= 0 or idx + step < len(all_ids)):
            if idx - step >= 0:
                picked.add(all_ids[idx - step])
                if len(picked) >= limit:
                    break
            if idx + step < len(all_ids):
                picked.add(all_ids[idx + step])
                if len(picked) >= limit:
                    break
            step += 1
        picked.discard(source_id)
        return picked

    def _load_merge_model_config(self):
        """Load merge-model selection from persisted settings."""
        try:
            setting = self.db.query(AppSetting).filter(AppSetting.key == "llm_merge_model").first()
            if setting and setting.value:
                self.merge_model_id = setting.value
            if self.merge_model_id.startswith("ollama"):
                ollama_setting = self.db.query(AppSetting).filter(AppSetting.key == "ollama_api_base").first()
                if ollama_setting and ollama_setting.value:
                    os.environ["OLLAMA_API_BASE"] = ollama_setting.value
            logger.info("Theme merge verification configured model: %s", self.merge_model_id)
        except Exception as exc:
            logger.warning("Could not load merge model setting; using default %s (%s)", self.merge_model_id, exc)

    def _init_llm_client(self):
        """Initialize LLMService for verification"""
        try:
            self.llm = LLMService(use_case="extraction")
            logger.info("LLMService initialized for theme merging (model=%s)", self.merge_model_id)
        except Exception as e:
            logger.warning(f"LLMService initialization failed: {e}")

    def _normalize_suggested_name(self, suggested_name: str | None) -> str | None:
        if not suggested_name or not suggested_name.strip():
            return None
        candidate = " ".join(suggested_name.strip().split())
        if canonical_theme_key(candidate) == UNKNOWN_THEME_KEY:
            return None
        return candidate[:200]

    def _maybe_with_for_update(self, query):
        """Apply row-level locking on databases that support it."""
        bind = self.db.get_bind()
        if bind is not None and bind.dialect.name != "sqlite":
            return query.with_for_update()
        return query

    @staticmethod
    def _canonical_pair_ids(cluster_a_id: int, cluster_b_id: int) -> tuple[int, int]:
        if cluster_a_id <= cluster_b_id:
            return cluster_a_id, cluster_b_id
        return cluster_b_id, cluster_a_id

    def _get_suggestion_for_pair(self, cluster_a_id: int, cluster_b_id: int) -> ThemeMergeSuggestion | None:
        pair_min_id, pair_max_id = self._canonical_pair_ids(cluster_a_id, cluster_b_id)
        suggestion = self.db.query(ThemeMergeSuggestion).filter(
            ThemeMergeSuggestion.pair_min_cluster_id == pair_min_id,
            ThemeMergeSuggestion.pair_max_cluster_id == pair_max_id,
        ).first()
        if suggestion:
            return suggestion
        # Legacy fallback before canonical pair columns were backfilled.
        return self.db.query(ThemeMergeSuggestion).filter(
            or_(
                and_(
                    ThemeMergeSuggestion.source_cluster_id == cluster_a_id,
                    ThemeMergeSuggestion.target_cluster_id == cluster_b_id,
                ),
                and_(
                    ThemeMergeSuggestion.source_cluster_id == cluster_b_id,
                    ThemeMergeSuggestion.target_cluster_id == cluster_a_id,
                ),
            )
        ).first()

    def _load_replay_result(
        self,
        suggestion: ThemeMergeSuggestion,
        *,
        idempotency_key: str,
    ) -> dict | None:
        if suggestion.approval_idempotency_key != idempotency_key:
            return None
        payload = suggestion.approval_result_json
        if not payload:
            return None
        try:
            result = json.loads(payload)
        except (TypeError, ValueError):
            logger.warning("Invalid approval_result_json for suggestion %s", suggestion.id)
            return None
        if not isinstance(result, dict):
            return None
        result["success"] = True
        result["idempotent_replay"] = True
        result["idempotency_key"] = idempotency_key
        return result

    def _build_success_result(
        self,
        *,
        source_name: str,
        target_name: str,
        constituents_merged: int,
        mentions_merged: int,
        idempotency_key: str | None,
        warning: str | None = None,
    ) -> dict:
        result = {
            "success": True,
            "source_name": source_name,
            "target_name": target_name,
            "constituents_merged": constituents_merged,
            "mentions_merged": mentions_merged,
        }
        if idempotency_key:
            result["idempotency_key"] = idempotency_key
        if warning:
            result["warning"] = warning
        return result

    def _get_theme_text(self, theme: ThemeCluster) -> str:
        """Generate text representation of theme for embedding"""
        return self.embedding_repo.build_theme_text(theme)

    def generate_theme_embedding(self, theme: ThemeCluster) -> Optional[np.ndarray]:
        """Generate embedding vector for a theme"""
        text = self._get_theme_text(theme)
        return self.embedding_engine.encode(text)

    def update_theme_embedding(self, theme: ThemeCluster) -> tuple[Optional[ThemeEmbedding], bool]:
        """Update or create embedding for a theme."""
        content_hash = self.embedding_repo.build_theme_content_hash(theme)
        existing = self.embedding_repo.get_for_cluster(theme.id)
        if not self.embedding_repo.embedding_needs_refresh(
            existing,
            content_hash=content_hash,
            embedding_model=self.EMBEDDING_MODEL,
            model_version=self.EMBEDDING_MODEL_VERSION,
        ):
            return existing, False

        embedding_array = self.generate_theme_embedding(theme)
        if embedding_array is None:
            return None, False

        # Serialize embedding as JSON list
        embedding_json = ThemeEmbeddingEngine.serialize(embedding_array)
        text = self._get_theme_text(theme)
        record = self.embedding_repo.upsert_for_theme(
            theme,
            embedding_json=embedding_json,
            embedding_text=text,
            embedding_model=self.EMBEDDING_MODEL,
            content_hash=content_hash,
            model_version=self.EMBEDDING_MODEL_VERSION,
            is_stale=False,
        )
        self.db.commit()
        return record, True

    def update_all_embeddings(self) -> dict:
        """Update embeddings for all active themes"""
        if self.embedding_engine.get_encoder() is None:
            return {"error": "Embedding model not available", "updated": 0, "skipped": 0}

        themes = self.db.query(ThemeCluster).filter(
            ThemeCluster.is_active == True
        ).all()

        updated = 0
        skipped = 0
        errors = 0

        for theme in themes:
            try:
                result, refreshed = self.update_theme_embedding(theme)
                if result and refreshed:
                    updated += 1
                elif result:
                    skipped += 1
                else:
                    errors += 1
            except Exception as e:
                logger.error(f"Error updating embedding for {theme.name}: {e}")
                errors += 1

        return {
            "total_themes": len(themes),
            "updated": updated,
            "skipped": skipped,
            "errors": errors,
        }

    def recompute_stale_embeddings(
        self,
        *,
        pipeline: str | None = None,
        batch_size: int = 100,
        max_batches: int = 20,
        on_batch: Callable[[dict], None] | None = None,
    ) -> dict:
        """
        Recompute only stale embeddings in bounded batches.

        Ordering oldest stale rows first plus single-run attempted-id exclusion
        prevents long-lived stale failures from starving newer stale rows.
        Failed rows are moved to the back of future runs by updating updated_at.
        """
        bounded_batch_size = max(1, int(batch_size or 1))
        bounded_max_batches = max(1, int(max_batches or 1))
        base_filters = [
            ThemeEmbedding.is_stale.is_(True),
            ThemeCluster.is_active == True,
        ]
        if pipeline:
            base_filters.append(ThemeCluster.pipeline == pipeline)

        stale_total_before = self.db.query(func.count(ThemeEmbedding.id)).join(
            ThemeCluster, ThemeCluster.id == ThemeEmbedding.theme_cluster_id
        ).filter(*base_filters).scalar() or 0

        processed = 0
        refreshed = 0
        unchanged = 0
        failed = 0
        batches_processed = 0
        attempted_cluster_ids: set[int] = set()
        failed_clusters: list[dict[str, object]] = []

        for batch_index in range(1, bounded_max_batches + 1):
            query = self.db.query(ThemeCluster).join(
                ThemeEmbedding, ThemeEmbedding.theme_cluster_id == ThemeCluster.id
            ).filter(*base_filters)
            if attempted_cluster_ids:
                query = query.filter(~ThemeCluster.id.in_(attempted_cluster_ids))
            clusters = query.order_by(
                func.coalesce(ThemeEmbedding.updated_at, ThemeEmbedding.created_at).asc(),
                ThemeEmbedding.id.asc(),
            ).limit(bounded_batch_size).all()
            if not clusters:
                break

            batches_processed += 1
            for cluster in clusters:
                attempted_cluster_ids.add(cluster.id)
                processed += 1
                try:
                    record, was_refreshed = self.update_theme_embedding(cluster)
                    if record is None:
                        failed += 1
                        failed_clusters.append({"theme_cluster_id": cluster.id, "theme": cluster.name})
                    elif was_refreshed:
                        refreshed += 1
                    else:
                        unchanged += 1
                except Exception as exc:
                    self.db.rollback()
                    failed += 1
                    failed_clusters.append(
                        {
                            "theme_cluster_id": cluster.id,
                            "theme": cluster.name,
                            "error": str(exc),
                        }
                    )
                    # Cross-run fairness: failed rows should not monopolize oldest-first batches.
                    # Keep row stale, but move its ordering timestamp forward.
                    self.db.query(ThemeEmbedding).filter(
                        ThemeEmbedding.theme_cluster_id == cluster.id
                    ).update(
                        {ThemeEmbedding.updated_at: datetime.utcnow()},
                        synchronize_session=False,
                    )
                    self.db.commit()
                    logger.error("Failed stale embedding recompute for cluster %s: %s", cluster.id, exc)

            stale_remaining = self.db.query(func.count(ThemeEmbedding.id)).join(
                ThemeCluster, ThemeCluster.id == ThemeEmbedding.theme_cluster_id
            ).filter(*base_filters).scalar() or 0

            if on_batch is not None:
                on_batch(
                    {
                        "batch": batch_index,
                        "batches_processed": batches_processed,
                        "processed": processed,
                        "refreshed": refreshed,
                        "unchanged": unchanged,
                        "failed": failed,
                        "stale_remaining": stale_remaining,
                        "stale_total_before": stale_total_before,
                    }
                )

        stale_remaining_after = self.db.query(func.count(ThemeEmbedding.id)).join(
            ThemeCluster, ThemeCluster.id == ThemeEmbedding.theme_cluster_id
        ).filter(*base_filters).scalar() or 0

        return {
            "pipeline": pipeline,
            "batch_size": bounded_batch_size,
            "max_batches": bounded_max_batches,
            "batches_processed": batches_processed,
            "stale_total_before": stale_total_before,
            "processed": processed,
            "refreshed": refreshed,
            "unchanged": unchanged,
            "failed": failed,
            "stale_remaining_after": stale_remaining_after,
            "has_more": stale_remaining_after > 0,
            "failed_clusters": failed_clusters[:25],
        }

    def _load_embedding(self, embedding_record: ThemeEmbedding) -> np.ndarray:
        """Load embedding from database record"""
        vector = ThemeEmbeddingEngine.deserialize(embedding_record.embedding)
        if vector is None:
            raise ValueError("Invalid embedding payload")
        return vector

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return ThemeEmbeddingEngine.cosine_similarity(a, b)

    def _embedding_record_needs_refresh(
        self,
        theme: ThemeCluster,
        record: ThemeEmbedding | None,
    ) -> bool:
        content_hash = self.embedding_repo.build_theme_content_hash(theme)
        return self.embedding_repo.embedding_needs_refresh(
            record,
            content_hash=content_hash,
            embedding_model=self.EMBEDDING_MODEL,
            model_version=self.EMBEDDING_MODEL_VERSION,
        )

    def find_similar_themes(
        self,
        theme_id: int,
        threshold: float = None
    ) -> list[dict]:
        """Find themes similar to the given theme"""
        if threshold is None:
            threshold = self.EMBEDDING_THRESHOLD

        source_theme = self.db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
        if not source_theme:
            return []

        # Get source theme embedding (refresh when missing/stale/outdated)
        source_embedding_record = self.embedding_repo.get_for_cluster(theme_id)
        if self._embedding_record_needs_refresh(source_theme, source_embedding_record):
            source_embedding_record, _ = self.update_theme_embedding(source_theme)
        if not source_embedding_record:
            return []

        source_embedding = self._load_embedding(source_embedding_record)

        other_themes = self.db.query(ThemeCluster).filter(
            ThemeCluster.id != theme_id,
            ThemeCluster.is_active == True,
        ).all()

        # Calculate similarities
        similar = []
        for theme in other_themes:
            other_record = self.embedding_repo.get_for_cluster(theme.id)
            if self._embedding_record_needs_refresh(theme, other_record):
                other_record, _ = self.update_theme_embedding(theme)
            if not other_record:
                continue

            try:
                other_vec = self._load_embedding(other_record)
            except Exception:
                continue

            similarity = self._cosine_similarity(source_embedding, other_vec)

            if similarity >= threshold:
                similar.append({
                    "theme_id": theme.id,
                    "name": theme.name,
                    "similarity": round(similarity, 4),
                    "aliases": theme.aliases,
                    "category": theme.category,
                })

        # Sort by similarity descending
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar

    def find_all_similar_pairs(
        self,
        threshold: float = None,
        pipeline: str = None,
        top_k: int | None = None,
        recall_sample_size: int = 0,
    ) -> list[dict]:
        """
        Find all pairs of similar themes above threshold.

        Only finds pairs within the same pipeline to prevent cross-pipeline merges.

        Args:
            threshold: Minimum similarity threshold
            pipeline: Filter to specific pipeline (if None, includes all but still requires same-pipeline matching)
        """
        if threshold is None:
            threshold = self.EMBEDDING_THRESHOLD

        # Load all embeddings with their theme pipeline info
        embeddings = self.embedding_repo.list_all()
        if len(embeddings) < 2:
            return []

        effective_top_k = max(1, int(top_k or self.CANDIDATE_TOP_K))
        # Build lookup with pipeline + lexical keys used for blocking.
        theme_data = []
        for emb in embeddings:
            if emb.is_stale:
                continue
            if (emb.embedding_model or "") != self.EMBEDDING_MODEL:
                continue
            if (emb.model_version or "") != self.EMBEDDING_MODEL_VERSION:
                continue
            theme = self.db.query(ThemeCluster).filter(
                ThemeCluster.id == emb.theme_cluster_id
            ).first()
            if theme and theme.is_active:
                theme_pipeline = theme.pipeline or "technical"
                # If pipeline filter is set, skip themes not in that pipeline
                if pipeline and theme_pipeline != pipeline:
                    continue
                theme_data.append({
                    "id": emb.theme_cluster_id,
                    "pipeline": theme_pipeline,
                    "canonical_key": (theme.canonical_key or "").strip().lower(),
                    "category": (theme.category or "").strip().lower(),
                    "vector": self._load_embedding(emb),
                })

        if len(theme_data) < 2:
            return []

        by_pipeline: dict[str, list[dict]] = {}
        for row in theme_data:
            by_pipeline.setdefault(row["pipeline"], []).append(row)

        pair_scores: dict[tuple[int, int], dict] = {}
        similarity_cache: dict[tuple[int, int], float] = {}
        compared_count = 0
        sampled_recall: float | None = None

        for pipeline_key, rows in by_pipeline.items():
            id_to_row = {int(row["id"]): row for row in rows}
            token_to_ids: dict[str, set[int]] = {}
            first_token_to_ids: dict[str, set[int]] = {}
            category_to_ids: dict[str, set[int]] = {}
            all_ids = sorted(id_to_row.keys())

            for row in rows:
                row_id = int(row["id"])
                key = str(row["canonical_key"] or "")
                tokens = [token for token in key.split("_") if token]
                row["lexical_tokens"] = set(tokens)
                row["first_token"] = tokens[0] if tokens else ""
                for token in row["lexical_tokens"]:
                    token_to_ids.setdefault(token, set()).add(row_id)
                if row["first_token"]:
                    first_token_to_ids.setdefault(row["first_token"], set()).add(row_id)
                if row["category"]:
                    category_to_ids.setdefault(str(row["category"]), set()).add(row_id)

            for source in rows:
                source_id = int(source["id"])
                source_tokens = set(source.get("lexical_tokens") or set())
                source_first = str(source.get("first_token") or "")
                source_category = str(source.get("category") or "")

                token_candidate_ids: set[int] = set()
                for token in source_tokens:
                    token_candidate_ids.update(token_to_ids.get(token, set()))
                first_candidate_ids: set[int] = set()
                if source_first:
                    first_candidate_ids.update(first_token_to_ids.get(source_first, set()))
                category_candidate_ids: set[int] = set()
                if source_category:
                    category_candidate_ids.update(category_to_ids.get(source_category, set()))

                block_cap = max(1, self.CANDIDATE_BLOCK_MAX // 3)
                candidate_ids = set()
                candidate_ids.update(self._cap_candidate_ids(token_candidate_ids, source_id=source_id, limit=block_cap))
                candidate_ids.update(self._cap_candidate_ids(first_candidate_ids, source_id=source_id, limit=block_cap))
                candidate_ids.update(self._cap_candidate_ids(category_candidate_ids, source_id=source_id, limit=block_cap))
                candidate_ids.discard(source_id)

                if not candidate_ids:
                    candidate_ids = self._fallback_neighbor_ids(
                        all_ids=all_ids,
                        source_id=source_id,
                        limit=self.CANDIDATE_BLOCK_MAX,
                    )

                scored_candidates: list[tuple[int, int]] = []
                for candidate_id in candidate_ids:
                    candidate_row = id_to_row.get(candidate_id)
                    if candidate_row is None:
                        continue
                    candidate_tokens = set(candidate_row.get("lexical_tokens") or set())
                    shared_tokens = len(source_tokens & candidate_tokens)
                    first_token_match = 1 if source_first and source_first == str(candidate_row.get("first_token") or "") else 0
                    category_match = 1 if source_category and source_category == str(candidate_row.get("category") or "") else 0
                    lexical_score = (shared_tokens * 4) + (first_token_match * 2) + category_match
                    scored_candidates.append((candidate_id, lexical_score))

                scored_candidates.sort(
                    key=lambda item: (-item[1], self._stable_candidate_rank(source_id, item[0]))
                )
                source_matches: list[tuple[int, float]] = []
                for candidate_id, _lexical_score in scored_candidates[: self.CANDIDATE_BLOCK_MAX]:
                    pair_key = (source_id, candidate_id) if source_id < candidate_id else (candidate_id, source_id)
                    similarity = similarity_cache.get(pair_key)
                    if similarity is None:
                        left = id_to_row[pair_key[0]]
                        right = id_to_row[pair_key[1]]
                        similarity = self._cosine_similarity(left["vector"], right["vector"])
                        similarity_cache[pair_key] = similarity
                        compared_count += 1
                    if similarity < threshold:
                        continue
                    source_matches.append((candidate_id, similarity))

                source_matches.sort(key=lambda item: item[1], reverse=True)
                for candidate_id, similarity in source_matches[:effective_top_k]:
                    pair_key = (source_id, candidate_id) if source_id < candidate_id else (candidate_id, source_id)
                    existing = pair_scores.get(pair_key)
                    payload = {
                        "theme1_id": pair_key[0],
                        "theme2_id": pair_key[1],
                        "similarity": round(similarity, 4),
                        "pipeline": pipeline_key,
                    }
                    if existing is None or payload["similarity"] > existing["similarity"]:
                        pair_scores[pair_key] = payload

        if recall_sample_size > 1:
            recall_denominator = 0
            recall_numerator = 0
            for pipeline_key, rows in by_pipeline.items():
                if len(rows) < 2:
                    continue
                sample_limit = min(len(rows), max(2, int(recall_sample_size)))
                sample_rows = sorted(rows, key=lambda row: int(row["id"]))[:sample_limit]
                sample_ids = {int(row["id"]) for row in sample_rows}
                for i in range(len(sample_rows)):
                    left_id = int(sample_rows[i]["id"])
                    for j in range(i + 1, len(sample_rows)):
                        right_id = int(sample_rows[j]["id"])
                        pair_key = (left_id, right_id) if left_id < right_id else (right_id, left_id)
                        similarity = similarity_cache.get(pair_key)
                        if similarity is None:
                            similarity = self._cosine_similarity(sample_rows[i]["vector"], sample_rows[j]["vector"])
                        if similarity >= threshold:
                            recall_denominator += 1
                            if pair_key in pair_scores:
                                recall_numerator += 1
            sampled_recall = 1.0 if recall_denominator == 0 else float(recall_numerator / recall_denominator)

        pairs = sorted(pair_scores.values(), key=lambda x: x["similarity"], reverse=True)

        self._last_pair_generation_metrics = {
            "themes_considered": len(theme_data),
            "compared_pairs": compared_count,
            "matched_pairs": len(pairs),
            "top_k": effective_top_k,
            "recall_sample_size": int(recall_sample_size or 0),
            "sampled_recall": sampled_recall,
        }
        logger.info(
            "find_all_similar_pairs optimized path: themes=%s compared_pairs=%s matched_pairs=%s top_k=%s sampled_recall=%s",
            len(theme_data),
            compared_count,
            len(pairs),
            effective_top_k,
            sampled_recall,
        )
        return pairs

    def verify_merge_with_llm(
        self,
        theme1: ThemeCluster,
        theme2: ThemeCluster,
        similarity: float
    ) -> dict:
        """Use LLM to verify if two themes should be merged"""
        if not self.llm:
            return {
                "error": "LLM not available",
                "should_merge": False,
                "confidence": 0.0,
            }

        # Rate limiting
        elapsed = time.time() - self._last_llm_request
        if elapsed < self._min_llm_interval:
            time.sleep(self._min_llm_interval - elapsed)

        prompt = MERGE_VERIFICATION_PROMPT.format(
            theme1_name=theme1.name,
            theme1_aliases=theme1.aliases or [],
            theme1_description=theme1.description or "No description",
            theme1_category=theme1.category or "Unknown",
            theme2_name=theme2.name,
            theme2_aliases=theme2.aliases or [],
            theme2_description=theme2.description or "No description",
            theme2_category=theme2.category or "Unknown",
            similarity=similarity,
        )

        try:
            logger.info(
                "Verifying merge with configured model %s (theme1=%s, theme2=%s, similarity=%.3f)",
                self.merge_model_id,
                theme1.id,
                theme2.id,
                similarity,
            )
            response = self.llm.completion_sync(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.merge_model_id,
                allow_fallbacks=False,
                temperature=0.1,
                max_tokens=500,
            )
            self._last_llm_request = time.time()

            # Get response content
            response_text = LLMService.extract_content(response)

            # Handle empty response
            if not response_text.strip():
                logger.warning("Empty response from LLM")
                return {
                    "error": "Empty response",
                    "should_merge": False,
                    "confidence": 0.0,
                }

            response_text = response_text.strip()

            # Clean up response (remove markdown code blocks if present)
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json) and last line (```)
                response_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            # Try to extract JSON from response if it contains other text
            if not response_text.startswith("{"):
                # Look for JSON object in the response
                json_match = re.search(r'\{[^{}]*"should_merge"[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)

            result = json.loads(response_text)

            return {
                "should_merge": result.get("should_merge", False),
                "confidence": result.get("confidence", 0.0),
                "relationship": result.get("relationship", "unknown"),
                "reasoning": result.get("reasoning", ""),
                "canonical_name": self._normalize_suggested_name(result.get("canonical_name")),
            }

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, response: {response_text[:200] if response_text else 'empty'}")
            return {
                "error": f"JSON parse error: {e}",
                "should_merge": False,
                "confidence": 0.0,
            }
        except LLMError as e:
            logger.error(f"LLM error for {theme1.name} vs {theme2.name}: {e}")
            return {
                "error": str(e),
                "should_merge": False,
                "confidence": 0.0,
            }
        except Exception as e:
            logger.error(f"Error verifying merge for {theme1.name} vs {theme2.name}: {e}")
            return {
                "error": str(e),
                "should_merge": False,
                "confidence": 0.0,
            }

    def create_merge_suggestion(
        self,
        source_id: int,
        target_id: int,
        similarity: float,
        llm_result: dict = None
    ) -> Optional[ThemeMergeSuggestion]:
        """Create a merge suggestion record"""
        pair_min_id, pair_max_id = self._canonical_pair_ids(source_id, target_id)
        existing = self._get_suggestion_for_pair(source_id, target_id)

        if existing:
            # Update existing suggestion
            existing.embedding_similarity = similarity
            existing.pair_min_cluster_id = pair_min_id
            existing.pair_max_cluster_id = pair_max_id
            if llm_result:
                existing.llm_confidence = llm_result.get("confidence")
                existing.llm_reasoning = llm_result.get("reasoning")
                existing.llm_relationship = llm_result.get("relationship")
                existing.suggested_canonical_name = self._normalize_suggested_name(
                    llm_result.get("canonical_name")
                )
            self.db.commit()
            return existing

        # Create new suggestion
        suggestion = ThemeMergeSuggestion(
            source_cluster_id=source_id,
            target_cluster_id=target_id,
            pair_min_cluster_id=pair_min_id,
            pair_max_cluster_id=pair_max_id,
            embedding_similarity=similarity,
            llm_confidence=llm_result.get("confidence") if llm_result else None,
            llm_reasoning=llm_result.get("reasoning") if llm_result else None,
            llm_relationship=llm_result.get("relationship") if llm_result else None,
            suggested_canonical_name=self._normalize_suggested_name(llm_result.get("canonical_name"))
            if llm_result
            else None,
            status="pending",
        )
        self.db.add(suggestion)
        try:
            self.db.commit()
        except IntegrityError:
            # Concurrent workers may race to insert the same pair. Re-load and update.
            self.db.rollback()
            existing = self._get_suggestion_for_pair(source_id, target_id)
            if not existing:
                raise
            existing.embedding_similarity = similarity
            existing.pair_min_cluster_id = pair_min_id
            existing.pair_max_cluster_id = pair_max_id
            if llm_result:
                existing.llm_confidence = llm_result.get("confidence")
                existing.llm_reasoning = llm_result.get("reasoning")
                existing.llm_relationship = llm_result.get("relationship")
                existing.suggested_canonical_name = self._normalize_suggested_name(
                    llm_result.get("canonical_name")
                )
            self.db.commit()
            return existing
        return suggestion

    def execute_merge(
        self,
        source_id: int,
        target_id: int,
        merge_type: str = "manual",
        suggestion: ThemeMergeSuggestion = None,
        expected_suggestion_status: str | None = None,
        final_suggestion_status: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Execute a theme merge: merge source INTO target"""
        try:
            if source_id == target_id:
                return {"error": "Source and target theme must be different", "success": False}
            ordered_theme_ids = sorted({source_id, target_id})
            locked_themes = self._maybe_with_for_update(
                self.db.query(ThemeCluster)
                .filter(ThemeCluster.id.in_(ordered_theme_ids))
                .order_by(ThemeCluster.id.asc())
            ).all()
            themes_by_id = {theme.id: theme for theme in locked_themes}
            source = themes_by_id.get(source_id)
            target = themes_by_id.get(target_id)

            if not source or not target:
                return {"error": "Theme not found", "success": False}

            if not source.is_active:
                return {"error": "Source theme already deactivated", "success": False}

            # Block cross-pipeline merges
            source_pipeline = source.pipeline or "technical"
            target_pipeline = target.pipeline or "technical"
            if source_pipeline != target_pipeline:
                return {
                    "error": f"Cross-pipeline merge not allowed: source is '{source_pipeline}', target is '{target_pipeline}'",
                    "success": False
                }

            locked_suggestion = None
            if suggestion:
                locked_suggestion = self._maybe_with_for_update(
                    self.db.query(ThemeMergeSuggestion).filter(ThemeMergeSuggestion.id == suggestion.id)
                ).first()
                if not locked_suggestion:
                    return {"error": "Suggestion not found", "success": False}
                if idempotency_key and locked_suggestion.approval_idempotency_key:
                    if locked_suggestion.approval_idempotency_key != idempotency_key:
                        return {"error": "Idempotency key mismatch for suggestion approval", "success": False}
                if expected_suggestion_status and locked_suggestion.status != expected_suggestion_status:
                    if idempotency_key:
                        replay = self._load_replay_result(locked_suggestion, idempotency_key=idempotency_key)
                        if replay:
                            return replay
                    return {
                        "error": f"Suggestion status changed to {locked_suggestion.status}",
                        "success": False,
                    }

            # Track counts
            constituents_merged = 0

            # 1. Merge aliases
            target_aliases = target.aliases or []
            if isinstance(target_aliases, str):
                target_aliases = json.loads(target_aliases)

            # Add source name and its aliases to target aliases
            if source.name not in target_aliases:
                target_aliases.append(source.name)

            source_aliases = source.aliases or []
            if isinstance(source_aliases, str):
                source_aliases = json.loads(source_aliases)

            for alias in source_aliases:
                if alias not in target_aliases:
                    target_aliases.append(alias)

            target.aliases = target_aliases

            # 2. Reassign constituents
            source_constituents = self.db.query(ThemeConstituent).filter(
                ThemeConstituent.theme_cluster_id == source_id
            ).all()

            for constituent in source_constituents:
                # Check if target already has this symbol
                existing = self.db.query(ThemeConstituent).filter(
                    ThemeConstituent.theme_cluster_id == target_id,
                    ThemeConstituent.symbol == constituent.symbol
                ).first()

                if existing:
                    # Update existing with higher values
                    existing.mention_count = max(
                        existing.mention_count or 0,
                        constituent.mention_count or 0
                    )
                    existing.confidence = max(
                        existing.confidence or 0,
                        constituent.confidence or 0
                    )
                    if constituent.first_mentioned_at:
                        if not existing.first_mentioned_at or constituent.first_mentioned_at < existing.first_mentioned_at:
                            existing.first_mentioned_at = constituent.first_mentioned_at
                    if constituent.last_mentioned_at:
                        if not existing.last_mentioned_at or constituent.last_mentioned_at > existing.last_mentioned_at:
                            existing.last_mentioned_at = constituent.last_mentioned_at
                    # Delete the source constituent
                    self.db.delete(constituent)
                else:
                    # Move constituent to target
                    constituent.theme_cluster_id = target_id
                constituents_merged += 1

            # 3. Reassign mentions
            self.db.query(ThemeMention).filter(
                ThemeMention.theme_cluster_id == source_id
            ).update(
                {ThemeMention.theme_cluster_id: target_id},
                synchronize_session=False
            )
            mentions_merged = self.db.query(ThemeMention).filter(
                ThemeMention.theme_cluster_id == target_id
            ).count()

            # 4. Deactivate source theme with explicit lifecycle transition audit.
            if (source.lifecycle_state or "active") != "retired":
                try:
                    apply_lifecycle_transition(
                        db=self.db,
                        theme=source,
                        to_state="retired",
                        actor="system",
                        job_name="theme_merge",
                        rule_version="lifecycle-v1",
                        reason=f"merged_into:{target_id}",
                        metadata={"merge_type": merge_type, "target_cluster_id": target_id},
                    )
                except ValueError as exc:
                    self.db.rollback()
                    return {"error": f"Lifecycle transition failed: {exc}", "success": False}
            else:
                source.is_active = False

            # 5. Delete source embedding
            self.db.query(ThemeEmbedding).filter(
                ThemeEmbedding.theme_cluster_id == source_id
            ).delete()

            # 6. Update target's first/last seen dates
            if source.first_seen_at:
                if not target.first_seen_at or source.first_seen_at < target.first_seen_at:
                    target.first_seen_at = source.first_seen_at
            if source.last_seen_at:
                if not target.last_seen_at or source.last_seen_at > target.last_seen_at:
                    target.last_seen_at = source.last_seen_at

            # 7. Create audit record
            history = ThemeMergeHistory(
                source_cluster_id=source_id,
                source_cluster_name=source.name,
                target_cluster_id=target_id,
                target_cluster_name=target.name,
                merge_type=merge_type,
                embedding_similarity=locked_suggestion.embedding_similarity if locked_suggestion else None,
                llm_confidence=locked_suggestion.llm_confidence if locked_suggestion else None,
                llm_reasoning=locked_suggestion.llm_reasoning if locked_suggestion else None,
                constituents_merged=constituents_merged,
                mentions_merged=mentions_merged,
            )
            self.db.add(history)
            self.db.flush()

            result_payload = self._build_success_result(
                source_name=source.name,
                target_name=target.name,
                constituents_merged=constituents_merged,
                mentions_merged=mentions_merged,
                idempotency_key=idempotency_key,
            )

            # 8. Atomically transition suggestion status if provided.
            if locked_suggestion:
                from_status = expected_suggestion_status or locked_suggestion.status
                to_status = final_suggestion_status or ("approved" if merge_type == "manual" else "auto_merged")
                status_filter = [
                    ThemeMergeSuggestion.id == locked_suggestion.id,
                    ThemeMergeSuggestion.status == from_status,
                ]
                if idempotency_key:
                    status_filter.append(
                        or_(
                            ThemeMergeSuggestion.approval_idempotency_key.is_(None),
                            ThemeMergeSuggestion.approval_idempotency_key == idempotency_key,
                        )
                    )
                update_payload = {
                    ThemeMergeSuggestion.status: to_status,
                    ThemeMergeSuggestion.reviewed_at: datetime.utcnow(),
                }
                if idempotency_key:
                    update_payload[ThemeMergeSuggestion.approval_idempotency_key] = idempotency_key
                    update_payload[ThemeMergeSuggestion.approval_result_json] = json.dumps(result_payload)
                updated_rows = self.db.query(ThemeMergeSuggestion).filter(*status_filter).update(
                    update_payload,
                    synchronize_session=False,
                )
                if updated_rows != 1:
                    self.db.rollback()
                    return {"error": "Suggestion status changed during merge", "success": False}

            self.db.commit()

            # 9. Update target embedding post-commit. Merge remains successful if refresh fails.
            embedding_warning = None
            try:
                self.update_theme_embedding(target)
            except Exception as exc:
                logger.exception("Target embedding refresh failed after merge %s -> %s: %s", source_id, target_id, exc)
                embedding_warning = str(exc)

            if embedding_warning:
                result_payload["warning"] = f"Merged successfully but target embedding refresh failed: {embedding_warning}"
            return result_payload
        except Exception as exc:
            self.db.rollback()
            logger.exception("Merge transaction failed for %s -> %s", source_id, target_id)
            return {"error": f"Merge transaction failed: {exc}", "success": False}

    def run_consolidation(self, dry_run: bool = True) -> dict:
        """
        Run the full theme consolidation pipeline.

        Steps:
        1. Update embeddings for all themes
        2. Find all similar pairs
        3. Verify with LLM
        4. Auto-merge high confidence, queue others for review

        Args:
            dry_run: If True, don't actually merge, just report what would happen

        Returns:
            Dict with consolidation results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "dry_run": dry_run,
            "embeddings_updated": 0,
            "pairs_found": 0,
            "llm_verified": 0,
            "auto_merged": 0,
            "queued_for_review": 0,
            "merge_details": [],
            "errors": [],
        }

        logger.info("=" * 60)
        logger.info(f"Theme Consolidation Pipeline (dry_run={dry_run})")
        logger.info("=" * 60)

        # Step 1: Update all embeddings
        logger.info("Step 1: Updating theme embeddings...")
        if self.embedding_engine.get_encoder() is not None:
            emb_result = self.update_all_embeddings()
            results["embeddings_updated"] = emb_result.get("updated", 0)
            logger.info(f"  Updated {results['embeddings_updated']} embeddings")
        else:
            results["errors"].append("Embedding model not available")
            logger.error("  Embedding model not available")
            return results

        # Step 2: Find similar pairs
        logger.info(f"Step 2: Finding similar pairs (threshold={self.EMBEDDING_THRESHOLD})...")
        pairs = self.find_all_similar_pairs(threshold=self.EMBEDDING_THRESHOLD)
        results["pairs_found"] = len(pairs)
        logger.info(f"  Found {len(pairs)} candidate pairs")

        if not pairs:
            logger.info("No similar pairs found")
            return results

        # Step 3: Verify with LLM and process
        logger.info("Step 3: LLM verification and processing...")
        for pair in pairs:
            theme1 = self.db.query(ThemeCluster).filter(
                ThemeCluster.id == pair["theme1_id"]
            ).first()
            theme2 = self.db.query(ThemeCluster).filter(
                ThemeCluster.id == pair["theme2_id"]
            ).first()

            if not theme1 or not theme2:
                continue

            if not theme1.is_active or not theme2.is_active:
                continue

            # Skip if already have a pending suggestion
            existing = self.db.query(ThemeMergeSuggestion).filter(
                or_(
                    and_(
                        ThemeMergeSuggestion.source_cluster_id == theme1.id,
                        ThemeMergeSuggestion.target_cluster_id == theme2.id
                    ),
                    and_(
                        ThemeMergeSuggestion.source_cluster_id == theme2.id,
                        ThemeMergeSuggestion.target_cluster_id == theme1.id
                    )
                ),
                ThemeMergeSuggestion.status.in_(["pending", "auto_merged", "approved"])
            ).first()

            if existing:
                continue

            similarity = pair["similarity"]

            # Verify with LLM
            llm_result = self.verify_merge_with_llm(theme1, theme2, similarity)
            results["llm_verified"] += 1

            if llm_result.get("error"):
                results["errors"].append(f"{theme1.name} vs {theme2.name}: {llm_result['error']}")
                continue

            detail = {
                "theme1": theme1.name,
                "theme2": theme2.name,
                "similarity": similarity,
                "llm_confidence": llm_result.get("confidence", 0),
                "llm_relationship": llm_result.get("relationship"),
                "should_merge": llm_result.get("should_merge", False),
                "action": None,
            }

            should_merge = llm_result.get("should_merge", False)
            confidence = llm_result.get("confidence", 0)

            if should_merge and similarity >= self.AUTO_MERGE_THRESHOLD and confidence >= self.LLM_CONFIDENCE_THRESHOLD:
                # Auto-merge
                detail["action"] = "auto_merge"
                if not dry_run:
                    # Determine which to keep (prefer the one with more constituents)
                    count1 = self.db.query(ThemeConstituent).filter(
                        ThemeConstituent.theme_cluster_id == theme1.id
                    ).count()
                    count2 = self.db.query(ThemeConstituent).filter(
                        ThemeConstituent.theme_cluster_id == theme2.id
                    ).count()

                    if count1 >= count2:
                        source_id, target_id = theme2.id, theme1.id
                    else:
                        source_id, target_id = theme1.id, theme2.id

                    suggestion = self.create_merge_suggestion(
                        source_id, target_id, similarity, llm_result
                    )
                    merge_result = self.execute_merge(
                        source_id,
                        target_id,
                        "auto",
                        suggestion,
                        expected_suggestion_status="pending",
                        final_suggestion_status="auto_merged",
                        idempotency_key=f"auto-merge-suggestion:{suggestion.id}",
                    )
                    if merge_result.get("success"):
                        results["auto_merged"] += 1
                        detail["merged_into"] = merge_result.get("target_name")
                    else:
                        results["errors"].append(f"Merge failed: {merge_result.get('error')}")
                else:
                    results["auto_merged"] += 1

            elif should_merge or (similarity >= self.EMBEDDING_THRESHOLD and confidence >= 0.5):
                # Queue for review
                detail["action"] = "queue_review"
                if not dry_run:
                    self.create_merge_suggestion(
                        theme1.id, theme2.id, similarity, llm_result
                    )
                results["queued_for_review"] += 1

            else:
                detail["action"] = "no_action"

            results["merge_details"].append(detail)
            logger.info(f"  {theme1.name} <-> {theme2.name}: {detail['action']} (sim={similarity:.3f}, conf={confidence:.2f})")

        logger.info("=" * 60)
        logger.info(f"Consolidation complete: {results['auto_merged']} auto-merged, {results['queued_for_review']} queued")
        logger.info("=" * 60)

        return results

    def get_merge_suggestions(
        self,
        status: str = None,
        limit: int = 50
    ) -> list[dict]:
        """Get merge suggestions from the queue"""
        query = self.db.query(ThemeMergeSuggestion)

        if status:
            query = query.filter(ThemeMergeSuggestion.status == status)

        suggestions = query.order_by(
            ThemeMergeSuggestion.embedding_similarity.desc()
        ).limit(limit).all()

        results = []
        for s in suggestions:
            source = self.db.query(ThemeCluster).filter(
                ThemeCluster.id == s.source_cluster_id
            ).first()
            target = self.db.query(ThemeCluster).filter(
                ThemeCluster.id == s.target_cluster_id
            ).first()

            if not source or not target:
                continue

            # Skip suggestions where either theme has been deactivated
            if not source.is_active or not target.is_active:
                continue

            # Skip cross-pipeline suggestions (can't merge technical with fundamental)
            if source.pipeline != target.pipeline:
                continue

            results.append({
                "id": s.id,
                # Canonical contract (frontend-facing)
                "source_theme_id": s.source_cluster_id,
                "source_theme_name": source.name,
                "source_aliases": source.aliases,
                "target_theme_id": s.target_cluster_id,
                "target_theme_name": target.name,
                "target_aliases": target.aliases,
                "similarity_score": s.embedding_similarity,
                "llm_confidence": s.llm_confidence,
                "relationship_type": s.llm_relationship,
                "reasoning": s.llm_reasoning,
                "suggested_name": s.suggested_canonical_name,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                # Legacy compatibility keys (deprecated)
                "source_cluster_id": s.source_cluster_id,
                "source_name": source.name,
                "target_cluster_id": s.target_cluster_id,
                "target_name": target.name,
                "embedding_similarity": s.embedding_similarity,
                "llm_reasoning": s.llm_reasoning,
                "llm_relationship": s.llm_relationship,
                "suggested_canonical_name": s.suggested_canonical_name,
            })

        return results

    def approve_suggestion(self, suggestion_id: int, idempotency_key: str | None = None) -> dict:
        """Approve and execute a merge suggestion"""
        operation_key = idempotency_key.strip()[:128] if idempotency_key and idempotency_key.strip() else None
        suggestion = self._maybe_with_for_update(
            self.db.query(ThemeMergeSuggestion).filter(ThemeMergeSuggestion.id == suggestion_id)
        ).first()

        if not suggestion:
            return {"error": "Suggestion not found", "success": False}

        if operation_key and suggestion.approval_idempotency_key and suggestion.approval_idempotency_key != operation_key:
            return {"error": "Idempotency key mismatch for suggestion approval", "success": False}

        if operation_key:
            replay = self._load_replay_result(suggestion, idempotency_key=operation_key)
            if replay:
                return replay

        if suggestion.status != "pending":
            return {"error": f"Suggestion already {suggestion.status}", "success": False}

        result = self.execute_merge(
            suggestion.source_cluster_id,
            suggestion.target_cluster_id,
            "manual",
            suggestion,
            expected_suggestion_status="pending",
            final_suggestion_status="approved",
            idempotency_key=operation_key,
        )

        if operation_key and not result.get("success"):
            refreshed = self.db.query(ThemeMergeSuggestion).filter(
                ThemeMergeSuggestion.id == suggestion_id
            ).first()
            if refreshed:
                replay_after_failure = self._load_replay_result(refreshed, idempotency_key=operation_key)
                if replay_after_failure:
                    return replay_after_failure

        return result

    def reject_suggestion(self, suggestion_id: int) -> dict:
        """Reject a merge suggestion"""
        updated_rows = self.db.query(ThemeMergeSuggestion).filter(
            ThemeMergeSuggestion.id == suggestion_id,
            ThemeMergeSuggestion.status == "pending",
        ).update(
            {
                ThemeMergeSuggestion.status: "rejected",
                ThemeMergeSuggestion.reviewed_at: datetime.utcnow(),
            },
            synchronize_session=False,
        )

        if updated_rows != 1:
            suggestion = self.db.query(ThemeMergeSuggestion).filter(
                ThemeMergeSuggestion.id == suggestion_id
            ).first()
            if not suggestion:
                return {"error": "Suggestion not found", "success": False}
            return {"error": f"Suggestion already {suggestion.status}", "success": False}

        self.db.commit()
        return {"success": True, "status": "rejected"}

    def get_merge_history(self, limit: int = 50) -> list[dict]:
        """Get merge history"""
        history = self.db.query(ThemeMergeHistory).order_by(
            ThemeMergeHistory.merged_at.desc()
        ).limit(limit).all()

        return [
            {
                "id": h.id,
                "source_name": h.source_cluster_name,
                "target_name": h.target_cluster_name,
                "merge_type": h.merge_type,
                "embedding_similarity": h.embedding_similarity,
                "llm_confidence": h.llm_confidence,
                "llm_reasoning": h.llm_reasoning,
                "constituents_merged": h.constituents_merged,
                "mentions_merged": h.mentions_merged,
                "merged_at": h.merged_at.isoformat() if h.merged_at else None,
                "merged_by": h.merged_by,
            }
            for h in history
        ]
