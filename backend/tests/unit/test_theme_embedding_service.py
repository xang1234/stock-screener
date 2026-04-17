from __future__ import annotations

import numpy as np

import app.services.theme_embedding_service as theme_embedding_service


class _FakeSentenceTransformer:
    init_calls: list[tuple[str, str]] = []

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        type(self).init_calls.append((model_name, device))

    def encode(
        self,
        payload,
        *,
        convert_to_numpy: bool = True,
        batch_size: int | None = None,
        normalize_embeddings: bool = False,
    ):
        if isinstance(payload, str):
            vector = np.array([float(len(payload)), 1.0], dtype=np.float32)
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            return vector

        rows = []
        for item in payload:
            vector = np.array([float(len(item)), 1.0], dtype=np.float32)
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            rows.append(vector)
        return np.vstack(rows)


def test_get_encoder_reuses_process_local_instance_per_model(monkeypatch):
    _FakeSentenceTransformer.init_calls.clear()
    monkeypatch.setattr(theme_embedding_service, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(theme_embedding_service, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(theme_embedding_service, "_ENCODER_CACHE", {}, raising=False)

    first = theme_embedding_service.ThemeEmbeddingEngine("all-MiniLM-L6-v2")
    second = theme_embedding_service.ThemeEmbeddingEngine("all-MiniLM-L6-v2")

    first_encoder = first.get_encoder()
    second_encoder = second.get_encoder()

    assert first_encoder is second_encoder
    assert _FakeSentenceTransformer.init_calls == [("all-MiniLM-L6-v2", "cpu")]


def test_encode_many_preserves_order_and_normalizes_vectors(monkeypatch):
    _FakeSentenceTransformer.init_calls.clear()
    monkeypatch.setattr(theme_embedding_service, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(theme_embedding_service, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(theme_embedding_service, "_ENCODER_CACHE", {}, raising=False)

    engine = theme_embedding_service.ThemeEmbeddingEngine("all-MiniLM-L6-v2")

    assert hasattr(engine, "encode_many")
    vectors = engine.encode_many(["A", "LONGER"], batch_size=8)

    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0))
    assert np.allclose(
        vectors[1],
        np.array([6.0, 1.0], dtype=np.float32) / np.linalg.norm(np.array([6.0, 1.0], dtype=np.float32)),
    )
