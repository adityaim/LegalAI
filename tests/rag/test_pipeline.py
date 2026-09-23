"""
tests/rag/test_pipeline.py
───────────────────────────
Integration tests for RAGPipeline (Module 5).

Uses MockEmbedder + MockVectorStore (zero external dependencies).
"""

import pytest

from modules.rag import RAGPipeline
from modules.rag.pipeline import PipelineConfig
from modules.rag.schema import RetrievalResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

NI_ACT_TEXT = (
    "Section 138 of the Negotiable Instruments Act deals with dishonour of cheque. "
    "The drawer shall be deemed to have committed an offence. "
    "Punishment includes imprisonment up to two years or fine. "
    "A legal notice must be sent within 30 days of dishonour. "
    "The accused must pay within 15 days of notice receipt."
)

IPC_TEXT = (
    "Section 420 of the Indian Penal Code deals with cheating. "
    "Whoever cheats and dishonestly induces delivery of property shall be punished. "
    "The punishment extends to seven years imprisonment. "
    "Fraudulent intent must be established from the inception of the transaction. "
    "The Supreme Court has interpreted this provision in many judgments."
)

FUSED_NI = f"[OCR-DOCUMENT]\n{NI_ACT_TEXT}\n[/OCR-DOCUMENT]"
FUSED_IPC = f"[OCR-DOCUMENT]\n{IPC_TEXT}\n[/OCR-DOCUMENT]"


class FakeFusedDoc:
    def __init__(self, text: str, fused: str):
        import uuid
        self.document_id = str(uuid.uuid4())
        self.fused_text = fused
        self._rag = text

    def for_rag(self) -> str:
        return self._rag


# ─── Ingest ───────────────────────────────────────────────────────────────────

class TestIngest:
    def test_ingest_returns_chunks(self):
        p = RAGPipeline()
        doc = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        chunks = p.ingest(doc)
        assert len(chunks) >= 1

    def test_collection_size_after_ingest(self):
        p = RAGPipeline()
        doc = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        chunks = p.ingest(doc)
        assert p.collection_size() == len(chunks)

    def test_ingest_multiple_docs(self):
        p = RAGPipeline()
        doc1 = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        doc2 = FakeFusedDoc(IPC_TEXT, FUSED_IPC)
        c1 = p.ingest(doc1)
        c2 = p.ingest(doc2)
        assert p.collection_size() == len(c1) + len(c2)

    def test_chunk_ids_unique_across_docs(self):
        p = RAGPipeline()
        doc1 = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        doc2 = FakeFusedDoc(IPC_TEXT, FUSED_IPC)
        c1 = p.ingest(doc1)
        c2 = p.ingest(doc2)
        all_ids = [c.chunk_id for c in c1 + c2]
        assert len(all_ids) == len(set(all_ids))

    def test_ingest_custom_collection(self):
        p = RAGPipeline()
        doc = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        p.ingest(doc, collection="case_001")
        assert p.collection_size("case_001") >= 1
        assert p.collection_size("legal_ai_default") == 0

    def test_ingest_empty_doc_returns_empty(self):
        p = RAGPipeline()
        doc = FakeFusedDoc("", "")
        chunks = p.ingest(doc)
        assert chunks == []


# ─── Query ────────────────────────────────────────────────────────────────────

class TestQuery:
    def setup_method(self):
        self.p = RAGPipeline()
        doc = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        self.p.ingest(doc)

    def test_query_returns_result(self):
        result = self.p.query("Section 138 dishonoured cheque")
        assert isinstance(result, RetrievalResult)

    def test_query_returns_chunks(self):
        result = self.p.query("legal notice within 30 days")
        assert len(result.chunks) >= 1

    def test_query_top_k_respected(self):
        result = self.p.query("Section 138", top_k=2)
        assert len(result.chunks) <= 2

    def test_query_scores_assigned(self):
        result = self.p.query("dishonour of cheque")
        for chunk in result.chunks:
            assert chunk.score is not None
            assert 0.0 <= chunk.score <= 1.0

    def test_retrieval_time_non_negative(self):
        result = self.p.query("test query")
        assert result.retrieval_time_s >= 0.0

    def test_query_string_preserved(self):
        q = "What is the punishment for Section 138 offence?"
        result = self.p.query(q)
        assert result.query == q

    def test_total_chunks_searched(self):
        result = self.p.query("cheque dishonour")
        assert result.total_chunks_searched == self.p.collection_size()

    def test_context_for_llm_non_empty(self):
        result = self.p.query("Section 138")
        assert len(result.context_for_llm) > 0
        assert "[CONTEXT]" in result.context_for_llm

    def test_empty_store_returns_no_chunks(self):
        p = RAGPipeline()  # fresh, empty store
        result = p.query("Section 138")
        assert result.chunks == []

    def test_mmr_disabled(self):
        result = self.p.query("Section 138", top_k=3, use_mmr=False)
        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) <= 3

    def test_min_score_filters_low_scores(self):
        result = self.p.query("completely irrelevant query xyz", min_score=0.99)
        # MockEmbedder hash-based; unlikely to hit 0.99 for irrelevant query
        # This just tests that the filter runs without error
        assert isinstance(result, RetrievalResult)


# ─── Collection management ────────────────────────────────────────────────────

class TestCollectionManagement:
    def test_clear_empties_collection(self):
        p = RAGPipeline()
        doc = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        p.ingest(doc)
        assert p.collection_size() >= 1
        p.clear()
        assert p.collection_size() == 0

    def test_collection_size_default_zero(self):
        p = RAGPipeline()
        assert p.collection_size() == 0

    def test_custom_collection_isolated(self):
        p = RAGPipeline()
        doc = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        p.ingest(doc, collection="case_a")
        assert p.collection_size("case_a") >= 1
        assert p.collection_size("case_b") == 0


# ─── End-to-end ───────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline(self):
        """Ingest two docs, query, validate context_for_llm format."""
        p = RAGPipeline()
        for text, fused in [(NI_ACT_TEXT, FUSED_NI), (IPC_TEXT, FUSED_IPC)]:
            p.ingest(FakeFusedDoc(text, fused))

        result = p.query("cheque dishonour legal notice", top_k=4)
        assert result.has_results()
        llm_ctx = result.context_for_llm
        assert "[CONTEXT]" in llm_ctx
        assert "[/CONTEXT]" in llm_ctx
        # At least one chunk from NI Act
        assert any("138" in c.text or "cheque" in c.text.lower() for c in result.chunks)

    def test_serialisation_after_query(self):
        p = RAGPipeline()
        p.ingest(FakeFusedDoc(NI_ACT_TEXT, FUSED_NI))
        result = p.query("Section 138")
        data = result.model_dump()
        restored = RetrievalResult.model_validate(data)
        assert restored.query == result.query
        assert len(restored.chunks) == len(result.chunks)

    def test_pipeline_config_chunk_size(self):
        """Smaller chunk size produces more chunks."""
        config_small = PipelineConfig(chunk_size=20, chunk_overlap=5, min_chunk_tokens=5)
        config_large = PipelineConfig(chunk_size=500, chunk_overlap=50, min_chunk_tokens=5)
        p_small = RAGPipeline(config=config_small)
        p_large = RAGPipeline(config=config_large)
        doc_s = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        doc_l = FakeFusedDoc(NI_ACT_TEXT, FUSED_NI)
        p_small.ingest(doc_s)
        p_large.ingest(doc_l)
        assert p_small.collection_size() >= p_large.collection_size()
