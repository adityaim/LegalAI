"""
modules/llm/schema.py
──────────────────────
Pydantic v2 schema for Module 6: LLM Legal Reasoning.

Contracts
─────────
• LegalQuery   — input to LegalReasoner.reason()
• LegalAnswer  — output of LegalReasoner.reason(); input to Module 7
• Citation     — individual evidence chunk cited in the answer

Design decisions
────────────────
• confidence (0–100) is the LLM's self-reported certainty, or a heuristic
  based on [UNCERTAIN] markers in the answer text.
• is_grounded = True iff at least one Citation was extracted from the
  retrieval context. An un-grounded answer is flagged for human review.
• review_required = True when confidence < confidence_threshold (70.0)
  or when is_grounded = False.
• answer_id is auto-generated UUID4.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ─── Citation ─────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """One retrieved chunk that supports (or is cited in) the LLM answer."""

    chunk_id: str = Field(..., description="chunk_id from the originating Chunk")
    document_id: str = Field(..., description="document_id of the parent FusedDocument")
    source_modality: str = Field("UNKNOWN", description="OCR | HTR | ASR | TYPED | UNKNOWN")
    text_excerpt: str = Field(..., description="First 200 chars of the chunk text")
    relevance_score: float = Field(..., ge=0.0, le=1.0)


# ─── LegalQuery ───────────────────────────────────────────────────────────────

class LegalQuery(BaseModel):
    """
    Input to LegalReasoner.reason() and POST /api/v1/llm/reason.

    fused_document_text  — FusedDocument.for_llm() output from Module 4.
    retrieval_context    — RetrievalResult.context_for_llm from Module 5.
    Both are optional (caller may pass an empty string if a module is bypassed).
    """

    query: str = Field(..., min_length=1, description="Natural language legal query")
    fused_document_text: str = Field("", description="Tagged document context from Module 4")
    retrieval_context: str = Field("", description="[CONTEXT]...[/CONTEXT] block from Module 5")
    language: str = Field("en", description="ISO 639-1 language code")
    detected_intent: str = Field("unknown", description="Intent from FusedDocument")
    max_tokens: int = Field(1024, ge=64, le=8192)


# ─── LegalAnswer ──────────────────────────────────────────────────────────────

class LegalAnswer(BaseModel):
    """
    Output of LegalReasoner.reason() and input to Module 7 (Response Generation).
    """

    answer_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID4 auto-generated per answer",
    )
    query: str = Field(...)
    answer_text: str = Field(..., min_length=1, description="Raw LLM answer text")
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=100.0)
    language: str = Field("en")
    detected_intent: str = Field("unknown")
    processing_time_s: float = Field(..., ge=0.0)
    llm_engine: str = Field(...)
    is_grounded: bool = Field(False)
    review_required: bool = Field(False)

    @model_validator(mode="after")
    def _derive_flags(self) -> "LegalAnswer":
        # is_grounded: True iff at least one citation extracted
        object.__setattr__(self, "is_grounded", len(self.citations) > 0)
        # review_required: low confidence OR no grounding
        object.__setattr__(
            self, "review_required",
            self.confidence < 70.0 or not self.is_grounded,
        )
        return self

    def summary(self) -> dict:
        """Compact dict for logging / health checks."""
        return {
            "answer_id": self.answer_id,
            "query_preview": self.query[:80],
            "confidence": self.confidence,
            "citations": len(self.citations),
            "is_grounded": self.is_grounded,
            "review_required": self.review_required,
            "llm_engine": self.llm_engine,
        }
