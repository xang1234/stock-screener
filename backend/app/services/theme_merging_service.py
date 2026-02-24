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
from typing import Optional

# Disable MPS/Metal before importing PyTorch to avoid fork() issues on macOS
# This must be set before any PyTorch imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_

from ..models.theme import (
    ThemeCluster,
    ThemeConstituent,
    ThemeMention,
    ThemeEmbedding,
    ThemeMergeSuggestion,
    ThemeMergeHistory,
)
from ..config import settings
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
    EMBEDDING_DIM = 384

    def __init__(self, db: Session):
        self.db = db
        self.embedding_engine = ThemeEmbeddingEngine(self.EMBEDDING_MODEL)
        self.embedding_repo = ThemeEmbeddingRepository(db)
        self.llm = None
        self._init_llm_client()

        # Rate limiting for LLM calls
        self._last_llm_request = 0
        self._min_llm_interval = 0.5  # seconds

    def _init_llm_client(self):
        """Initialize LLMService for verification"""
        try:
            self.llm = LLMService(use_case="extraction")
            logger.info("LLMService initialized for theme merging")
        except Exception as e:
            logger.warning(f"LLMService initialization failed: {e}")

    def _normalize_suggested_name(self, suggested_name: str | None) -> str | None:
        if not suggested_name or not suggested_name.strip():
            return None
        candidate = " ".join(suggested_name.strip().split())
        if canonical_theme_key(candidate) == UNKNOWN_THEME_KEY:
            return None
        return candidate[:200]

    def _get_theme_text(self, theme: ThemeCluster) -> str:
        """Generate text representation of theme for embedding"""
        return self.embedding_repo.build_theme_text(theme)

    def generate_theme_embedding(self, theme: ThemeCluster) -> Optional[np.ndarray]:
        """Generate embedding vector for a theme"""
        text = self._get_theme_text(theme)
        return self.embedding_engine.encode(text)

    def update_theme_embedding(self, theme: ThemeCluster) -> Optional[ThemeEmbedding]:
        """Update or create embedding for a theme"""
        embedding_array = self.generate_theme_embedding(theme)
        if embedding_array is None:
            return None

        # Serialize embedding as JSON list
        embedding_json = ThemeEmbeddingEngine.serialize(embedding_array)
        text = self._get_theme_text(theme)
        record = self.embedding_repo.upsert_for_theme(
            theme,
            embedding_json=embedding_json,
            embedding_text=text,
            embedding_model=self.EMBEDDING_MODEL,
        )
        self.db.commit()
        return record

    def update_all_embeddings(self) -> dict:
        """Update embeddings for all active themes"""
        if self.embedding_engine.get_encoder() is None:
            return {"error": "Embedding model not available", "updated": 0}

        themes = self.db.query(ThemeCluster).filter(
            ThemeCluster.is_active == True
        ).all()

        updated = 0
        errors = 0

        for theme in themes:
            try:
                result = self.update_theme_embedding(theme)
                if result:
                    updated += 1
                else:
                    errors += 1
            except Exception as e:
                logger.error(f"Error updating embedding for {theme.name}: {e}")
                errors += 1

        return {
            "total_themes": len(themes),
            "updated": updated,
            "errors": errors,
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

    def find_similar_themes(
        self,
        theme_id: int,
        threshold: float = None
    ) -> list[dict]:
        """Find themes similar to the given theme"""
        if threshold is None:
            threshold = self.EMBEDDING_THRESHOLD

        # Get source theme embedding
        source_embedding_record = self.embedding_repo.get_for_cluster(theme_id)

        if not source_embedding_record:
            # Try to generate it
            theme = self.db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
            if theme:
                self.update_theme_embedding(theme)
                source_embedding_record = self.embedding_repo.get_for_cluster(theme_id)

        if not source_embedding_record:
            return []

        source_embedding = self._load_embedding(source_embedding_record)

        # Get all other embeddings
        other_embeddings = [
            row for row in self.embedding_repo.list_all() if row.theme_cluster_id != theme_id
        ]

        # Calculate similarities
        similar = []
        for other in other_embeddings:
            other_vec = self._load_embedding(other)
            similarity = self._cosine_similarity(source_embedding, other_vec)

            if similarity >= threshold:
                theme = self.db.query(ThemeCluster).filter(
                    ThemeCluster.id == other.theme_cluster_id
                ).first()
                if theme and theme.is_active:
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
        pipeline: str = None
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

        # Build lookup with pipeline info
        theme_data = []
        for emb in embeddings:
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
                    "vector": self._load_embedding(emb),
                })

        if len(theme_data) < 2:
            return []

        # Calculate pairwise similarities - ONLY within same pipeline
        pairs = []
        n = len(theme_data)
        for i in range(n):
            for j in range(i + 1, n):
                # Block cross-pipeline pairs
                if theme_data[i]["pipeline"] != theme_data[j]["pipeline"]:
                    continue

                similarity = self._cosine_similarity(
                    theme_data[i]["vector"],
                    theme_data[j]["vector"]
                )
                if similarity >= threshold:
                    pairs.append({
                        "theme1_id": theme_data[i]["id"],
                        "theme2_id": theme_data[j]["id"],
                        "similarity": round(similarity, 4),
                        "pipeline": theme_data[i]["pipeline"],
                    })

        # Sort by similarity descending
        pairs.sort(key=lambda x: x["similarity"], reverse=True)
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
            response = self.llm.completion_sync(
                messages=[
                    {"role": "user", "content": prompt}
                ],
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
        # Check if suggestion already exists
        existing = self.db.query(ThemeMergeSuggestion).filter(
            or_(
                and_(
                    ThemeMergeSuggestion.source_cluster_id == source_id,
                    ThemeMergeSuggestion.target_cluster_id == target_id
                ),
                and_(
                    ThemeMergeSuggestion.source_cluster_id == target_id,
                    ThemeMergeSuggestion.target_cluster_id == source_id
                )
            )
        ).first()

        if existing:
            # Update existing suggestion
            existing.embedding_similarity = similarity
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
        self.db.commit()
        return suggestion

    def execute_merge(
        self,
        source_id: int,
        target_id: int,
        merge_type: str = "manual",
        suggestion: ThemeMergeSuggestion = None
    ) -> dict:
        """Execute a theme merge: merge source INTO target"""
        source = self.db.query(ThemeCluster).filter(ThemeCluster.id == source_id).first()
        target = self.db.query(ThemeCluster).filter(ThemeCluster.id == target_id).first()

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

        # Track counts
        constituents_merged = 0
        mentions_merged = 0

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
            embedding_similarity=suggestion.embedding_similarity if suggestion else None,
            llm_confidence=suggestion.llm_confidence if suggestion else None,
            llm_reasoning=suggestion.llm_reasoning if suggestion else None,
            constituents_merged=constituents_merged,
            mentions_merged=mentions_merged,
        )
        self.db.add(history)

        # 8. Update suggestion status if provided
        if suggestion:
            suggestion.status = "approved" if merge_type == "manual" else "auto_merged"
            suggestion.reviewed_at = datetime.utcnow()

        self.db.commit()

        # 9. Update target embedding
        self.update_theme_embedding(target)

        return {
            "success": True,
            "source_name": source.name,
            "target_name": target.name,
            "constituents_merged": constituents_merged,
            "mentions_merged": mentions_merged,
        }

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
                        source_id, target_id, "auto", suggestion
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
                "source_cluster_id": s.source_cluster_id,
                "source_name": source.name,
                "source_aliases": source.aliases,
                "target_cluster_id": s.target_cluster_id,
                "target_name": target.name,
                "target_aliases": target.aliases,
                "embedding_similarity": s.embedding_similarity,
                "llm_confidence": s.llm_confidence,
                "llm_reasoning": s.llm_reasoning,
                "llm_relationship": s.llm_relationship,
                "suggested_canonical_name": s.suggested_canonical_name,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            })

        return results

    def approve_suggestion(self, suggestion_id: int) -> dict:
        """Approve and execute a merge suggestion"""
        suggestion = self.db.query(ThemeMergeSuggestion).filter(
            ThemeMergeSuggestion.id == suggestion_id
        ).first()

        if not suggestion:
            return {"error": "Suggestion not found", "success": False}

        if suggestion.status != "pending":
            return {"error": f"Suggestion already {suggestion.status}", "success": False}

        result = self.execute_merge(
            suggestion.source_cluster_id,
            suggestion.target_cluster_id,
            "manual",
            suggestion
        )

        return result

    def reject_suggestion(self, suggestion_id: int) -> dict:
        """Reject a merge suggestion"""
        suggestion = self.db.query(ThemeMergeSuggestion).filter(
            ThemeMergeSuggestion.id == suggestion_id
        ).first()

        if not suggestion:
            return {"error": "Suggestion not found", "success": False}

        suggestion.status = "rejected"
        suggestion.reviewed_at = datetime.utcnow()
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
