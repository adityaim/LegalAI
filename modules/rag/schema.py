"""
modules/rag/schema.py
──────────────────────
Pydantic v2 schema for Module 5: RAG Ingestion & Retrieval.

All inter-module contracts are defined here and used by:
  • chunker.py        — produces Chunk objects
  • ingestor.py       — upserts Chunk objects to vector store
  • retriever.py      — returns RetrievalResult
  • pipeline.py       — orchestrates everything
  • rag_router.py     — serialises to/from JSON for FastAPI

Design decisions
─────────────────
• Chunk is immutable after creation (all fields required at construction time).
• score is Optional[float] — None until retrieval assigns a similarity score.
• RAGQuery wraps query text + retrieval parameters for the REST layer.
• RetrievalResult includes both raw chunks and a formatted context string
  ready to be passed to Module 6 (LLM Legal Reasoning).
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ─── Source tags (mirrors fusion modality sources) ────────────────────────────

ModalitySource = str   # "OCR" | "HTR" | "ASR" | "TYPED" | "UNKNOWN"


# ─── Chunk ────────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """
    A single text chunk produced by the chunker and stored in the vector store.

    chunk_id   — UUID4, assigned at chunk creation.
    document_id — UUID4 from the originating FusedDocument.
    source_modality — which modality block this chunk came from.
    text        — the raw chunk text (may contain [REVIEW] markers).
    chunk_index — 0-based position of this chunk in the document.
    start_char  — character offset in the for_rag() text where this chunk starts.
    end_char    — character offset where it ends (exclusive).
    has_review_flag — True if text contains a [REVIEW] marker.
    score       — cosine similarity score [0, 1], set after retrieval (None before).
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str = Field(..., description="UUID of the source FusedDocument")
    source_modality: ModalitySource = Field("UNKNOWN")
    text: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    has_review_flag: bool = Field(False)
    score: Optional[float] = Field(None, ge=0.0, le=1.0)

    def is_low_confidence(self) -> bool:
        """True if this chunk was flagged for human review."""
        return self.has_review_flag

    def preview(self, chars: int = 120) -> str:
        """Return a short preview of the chunk text."""
        text = self.text.replace("\n", " ")
        return text[:chars] + "..." if len(text) > chars else text


# ─── Query ────────────────────────────────────────────────────────────────────

class RAGQuery(BaseModel):
    """
    Input to RAGPipeline.query() and POST /api/v1/rag/query.
    """
    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: int = Field(5, ge=1, le=50, description="Max chunks to return")
    collection: str = Field("legal_ai_default", description="Vector store collection name")
    use_mmr: bool = Field(True, description="Apply MMR reranking to diversify results")
    mmr_lambda: float = Field(
        0.5, ge=0.0, le=1.0,
        description="MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity"
    )
    min_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Minimum similarity score threshold to include a chunk"
    )


# ─── Retrieval result ─────────────────────────────────────────────────────────

class RetrievalResult(BaseModel):
    """
    Output of RAGPipeline.query().

    chunks          — ranked list of Chunk objects with scores.
    query           — the original query string.
    total_chunks_searched — how many chunks were in the collection.
    context_for_llm — pre-formatted string for Module 6 LLM context window.
    retrieval_time_s — wall-clock time.
    """

    chunks: list[Chunk] = Field(default_factory=list)
    query: str = Field(...)
    total_chunks_searched: int = Field(0, ge=0)
    retrieval_time_s: float = Field(..., ge=0.0)

    @property
    def context_for_llm(self) -> str:
        """
        Format retrieved chunks into a [CONTEXT] block for Module 6.

        Each chunk is labelled with its source modality and score.
        [REVIEW] markers are preserved so the LLM knows which spans to treat
        with lower confidence.
        """
        if not self.chunks:
            return "[CONTEXT]\n(no relevant context found)\n[/CONTEXT]"

        parts = ["[CONTEXT]"]
        for i, chunk in enumerate(self.chunks, 1):
            score_str = f"{chunk.score:.3f}" if chunk.score is not None else "n/a"
            parts.append(
                f"[{i}] source={chunk.source_modality} score={score_str}\n"
                f"{chunk.text}"
            )
        parts.append("[/CONTEXT]")
        return "\n\n".join(parts)

    def has_results(self) -> bool:
        return len(self.chunks) > 0

    def top_chunk(self) -> Chunk | None:
        return self.chunks[0] if self.chunks else None
