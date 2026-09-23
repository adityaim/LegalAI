"""
tests/rag/test_schema.py
─────────────────────────
Unit tests for Module 5 RAG schema: Chunk, RAGQuery, RetrievalResult.
"""

import pytest
from pydantic import ValidationError

from modules.rag.schema import Chunk, RAGQuery, RetrievalResult


# ─── Chunk ────────────────────────────────────────────────────────────────────

class TestChunk:
    def make(self, **kw) -> Chunk:
        defaults = dict(
            document_id="doc-001",
            source_modality="OCR",
            text="Section 138 of the NI Act deals with dishonour of cheque.",
            chunk_index=0,
            start_char=0,
            end_char=60,
        )
        defaults.update(kw)
        return Chunk(**defaults)

    def test_valid_chunk(self):
        c = self.make()
        assert c.document_id == "doc-001"
        assert c.source_modality == "OCR"
        assert len(c.chunk_id) == 36  # UUID4

    def test_auto_chunk_id(self):
        c1 = self.make()
        c2 = self.make()
        assert c1.chunk_id != c2.chunk_id

    def test_score_defaults_none(self):
        c = self.make()
        assert c.score is None

    def test_score_valid_range(self):
        c = self.make(score=0.85)
        assert c.score == 0.85

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            self.make(score=1.5)

    def test_negative_score_rejected(self):
        with pytest.raises(ValidationError):
            self.make(score=-0.1)

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            self.make(text="")

    def test_is_low_confidence_false(self):
        c = self.make(has_review_flag=False)
        assert c.is_low_confidence() is False

    def test_is_low_confidence_true(self):
        c = self.make(has_review_flag=True)
        assert c.is_low_confidence() is True

    def test_preview_short_text(self):
        c = self.make(text="Short text.")
        assert c.preview() == "Short text."

    def test_preview_long_text(self):
        long_text = "word " * 100
        c = self.make(text=long_text)
        preview = c.preview(50)
        assert preview.endswith("...")
        assert len(preview) <= 53  # 50 chars + "..."

    def test_serialisation_round_trip(self):
        c = self.make(score=0.92, has_review_flag=True)
        data = c.model_dump()
        restored = Chunk.model_validate(data)
        assert restored.chunk_id == c.chunk_id
        assert restored.score == 0.92


# ─── RAGQuery ─────────────────────────────────────────────────────────────────

class TestRAGQuery:
    def test_valid_defaults(self):
        q = RAGQuery(query="What is Section 138?")
        assert q.top_k == 5
        assert q.use_mmr is True
        assert q.mmr_lambda == 0.5
        assert q.min_score == 0.0
        assert q.collection == "legal_ai_default"

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            RAGQuery(query="")

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            RAGQuery(query="test", top_k=0)
        with pytest.raises(ValidationError):
            RAGQuery(query="test", top_k=51)

    def test_mmr_lambda_bounds(self):
        with pytest.raises(ValidationError):
            RAGQuery(query="test", mmr_lambda=1.5)
        with pytest.raises(ValidationError):
            RAGQuery(query="test", mmr_lambda=-0.1)

    def test_custom_collection(self):
        q = RAGQuery(query="test", collection="case_456")
        assert q.collection == "case_456"


# ─── RetrievalResult ──────────────────────────────────────────────────────────

def make_chunk(text: str, score: float = 0.9, modality: str = "OCR") -> Chunk:
    return Chunk(
        document_id="doc-001",
        source_modality=modality,
        text=text,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        score=score,
    )


class TestRetrievalResult:
    def make(self, chunks=None) -> RetrievalResult:
        return RetrievalResult(
            chunks=chunks or [make_chunk("Section 138 NI Act.")],
            query="What is Section 138?",
            total_chunks_searched=10,
            retrieval_time_s=0.05,
        )

    def test_has_results_true(self):
        assert self.make().has_results() is True

    def test_has_results_false(self):
        r = RetrievalResult(chunks=[], query="x", total_chunks_searched=0, retrieval_time_s=0.0)
        assert r.has_results() is False

    def test_top_chunk_returns_first(self):
        c1 = make_chunk("First chunk", score=0.95)
        c2 = make_chunk("Second chunk", score=0.80)
        r = self.make(chunks=[c1, c2])
        assert r.top_chunk().text == "First chunk"

    def test_top_chunk_empty(self):
        r = RetrievalResult(chunks=[], query="x", total_chunks_searched=0, retrieval_time_s=0.0)
        assert r.top_chunk() is None

    def test_context_for_llm_has_context_tag(self):
        r = self.make()
        ctx = r.context_for_llm
        assert "[CONTEXT]" in ctx
        assert "[/CONTEXT]" in ctx

    def test_context_for_llm_contains_chunk_text(self):
        c = make_chunk("This is important legal text.")
        r = self.make(chunks=[c])
        assert "important legal text" in r.context_for_llm

    def test_context_for_llm_includes_score(self):
        c = make_chunk("Legal text.", score=0.876)
        r = self.make(chunks=[c])
        assert "0.876" in r.context_for_llm

    def test_context_for_llm_no_results(self):
        r = RetrievalResult(chunks=[], query="x", total_chunks_searched=0, retrieval_time_s=0.0)
        assert "no relevant context found" in r.context_for_llm

    def test_serialisation_round_trip(self):
        r = self.make()
        data = r.model_dump()
        restored = RetrievalResult.model_validate(data)
        assert restored.query == r.query
        assert len(restored.chunks) == len(r.chunks)
