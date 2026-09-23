"""
tests/rag/test_embedder.py
───────────────────────────
Unit tests for the MockEmbedder (Module 5).

Real embedders (SentenceTransformer, Gemini) are tested only when explicitly
opted-in via environment variables (same pattern as ASR tests skipping Whisper).
"""

import math
import pytest

from modules.rag.embedder import MockEmbedder


class TestMockEmbedder:
    def test_embed_single(self):
        emb = MockEmbedder()
        result = emb.embed(["Hello world"])
        assert len(result) == 1
        assert len(result[0]) == 128  # default dim

    def test_embed_batch(self):
        emb = MockEmbedder()
        texts = ["Section 138", "IPC 420", "High Court judgment"]
        result = emb.embed(texts)
        assert len(result) == 3
        assert all(len(v) == 128 for v in result)

    def test_dimension_property(self):
        emb = MockEmbedder(dim=64)
        assert emb.dimension == 64

    def test_custom_dim(self):
        emb = MockEmbedder(dim=256)
        result = emb.embed(["test"])
        assert len(result[0]) == 256

    def test_deterministic(self):
        emb = MockEmbedder()
        r1 = emb.embed(["Section 138 NI Act"])[0]
        r2 = emb.embed(["Section 138 NI Act"])[0]
        assert r1 == r2

    def test_different_texts_different_vectors(self):
        emb = MockEmbedder()
        v1 = emb.embed(["Section 138"])[0]
        v2 = emb.embed(["IPC 420"])[0]
        assert v1 != v2

    def test_vectors_normalised(self):
        emb = MockEmbedder()
        vec = emb.embed(["test normalisation"])[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_model_name_contains_dim(self):
        emb = MockEmbedder(dim=128)
        assert "128" in emb.model_name

    def test_empty_text_embeds(self):
        emb = MockEmbedder()
        result = emb.embed([" "])
        assert len(result) == 1
        assert len(result[0]) == 128

    def test_unicode_text(self):
        emb = MockEmbedder()
        result = emb.embed(["Section 138 - \u0928\u093f \u0905\u0927\u093f\u0928\u093f\u092f\u092e"])
        assert len(result) == 1
        assert len(result[0]) == 128
