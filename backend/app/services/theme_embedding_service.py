"""Shared embedding primitives for theme extraction and consolidation flows."""
from __future__ import annotations

import importlib.util
import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..models.theme import ThemeCluster, ThemeEmbedding

logger = logging.getLogger(__name__)

SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
SentenceTransformer = None


class ThemeEmbeddingEngine:
    """Shared embedding model wrapper + vector math helpers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._encoder = None

    def get_encoder(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        if self._encoder is not None:
            return self._encoder

        global SentenceTransformer
        try:
            if SentenceTransformer is None:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer

                SentenceTransformer = _SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name, device="cpu")
            return self._encoder
        except Exception as exc:
            logger.warning("Embedding encoder unavailable for model %s: %s", self.model_name, exc)
            self._encoder = None
            return None

    def encode(self, text: str) -> Optional[np.ndarray]:
        encoder = self.get_encoder()
        if encoder is None:
            return None
        try:
            return np.array(encoder.encode(text, convert_to_numpy=True))
        except Exception as exc:
            logger.warning("Failed to encode embedding text for model %s: %s", self.model_name, exc)
            return None

    @staticmethod
    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    @staticmethod
    def serialize(vector: np.ndarray) -> str:
        return json.dumps(vector.tolist())

    @staticmethod
    def deserialize(payload: str | None) -> Optional[np.ndarray]:
        if not payload:
            return None
        try:
            return np.array(json.loads(payload))
        except Exception:
            return None


class ThemeEmbeddingRepository:
    """Shared persistence/read helpers for theme embeddings."""

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def build_theme_text(theme: ThemeCluster) -> str:
        parts = [theme.name]
        if theme.aliases and isinstance(theme.aliases, list):
            aliases = [str(alias) for alias in theme.aliases if str(alias).strip()]
            if aliases:
                parts.append(f"Also known as: {', '.join(aliases)}")
        if theme.description:
            parts.append(theme.description)
        if theme.category:
            parts.append(f"Category: {theme.category}")
        return " | ".join(parts)

    def get_for_cluster(self, theme_cluster_id: int) -> Optional[ThemeEmbedding]:
        return self.db.query(ThemeEmbedding).filter(
            ThemeEmbedding.theme_cluster_id == theme_cluster_id
        ).first()

    def get_by_cluster_ids(
        self,
        cluster_ids: list[int],
        *,
        embedding_model: str | None = None,
        freshness_cutoff: datetime | None = None,
    ) -> dict[int, ThemeEmbedding]:
        if not cluster_ids:
            return {}
        query = self.db.query(ThemeEmbedding).filter(ThemeEmbedding.theme_cluster_id.in_(cluster_ids))
        if embedding_model:
            query = query.filter(ThemeEmbedding.embedding_model == embedding_model)
        if freshness_cutoff:
            query = query.filter(or_(ThemeEmbedding.updated_at.is_(None), ThemeEmbedding.updated_at >= freshness_cutoff))
        rows = query.all()
        return {row.theme_cluster_id: row for row in rows}

    def list_all(self) -> list[ThemeEmbedding]:
        return self.db.query(ThemeEmbedding).all()

    def upsert_for_theme(
        self,
        theme: ThemeCluster,
        *,
        embedding_json: str,
        embedding_text: str,
        embedding_model: str,
    ) -> ThemeEmbedding:
        existing = self.get_for_cluster(theme.id)
        if existing:
            existing.embedding = embedding_json
            existing.embedding_text = embedding_text
            existing.embedding_model = embedding_model
            existing.updated_at = datetime.utcnow()
            return existing

        record = ThemeEmbedding(
            theme_cluster_id=theme.id,
            embedding=embedding_json,
            embedding_model=embedding_model,
            embedding_text=embedding_text,
        )
        self.db.add(record)
        return record
