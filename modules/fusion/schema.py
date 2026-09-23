"""
modules/fusion/schema.py
─────────────────────────
Pydantic v2 output schema for Module 4: Text Fusion & Pre-processing.

FusedDocument is the canonical contract between Module 4 and all
downstream modules (Module 5 RAG Ingestion, Module 6 LLM Reasoning).

Design spec (Section 4.4):
  - One ModalityBlock per input modality (OCR / HTR / ASR / TYPED)
  - fused_text: tagged composite — each block wrapped in [SOURCE]…[/SOURCE]
  - Cleaned of control chars, normalised whitespace, deduped near-sentences
  - primary_language: most frequent language across all inputs
  - detected_intent: inherits from ASR if present; otherwise heuristic
  - review_flags: list of (modality, text) pairs requiring human review
  - for_rag(): plain-text view with tags stripped — ready for vector embedding
  - for_llm(): full tagged text with metadata header — ready for LLM context window
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ─── Modality source tags ─────────────────────────────────────────────────────

ModalitySource = Literal["OCR", "HTR", "ASR", "TYPED"]

LegalIntent = Literal["question", "instruction", "narration", "unknown"]


# ─── Per-modality block ───────────────────────────────────────────────────────

class ReviewFlag(BaseModel):
    """A single piece of text flagged for human review from a modality."""
    modality: ModalitySource
    text: str
    reason: str = "low_confidence"


class ModalityBlock(BaseModel):
    """
    One input modality's contribution to the fused document.

    Stores the cleaned, normalised text with its source tag and metadata.
    """
    source: ModalitySource
    text: str = Field("", description="Cleaned, normalised text from this modality")
    language: str = Field("en", description="Detected language ISO 639-1")
    word_count: int = Field(0, ge=0)
    has_review_flags: bool = Field(False, description="True if any text needs human review")

    @model_validator(mode="after")
    def compute_word_count(self) -> "ModalityBlock":
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())
        return self


# ─── Top-level fused document ─────────────────────────────────────────────────

class FusedDocument(BaseModel):
    """
    The canonical output of TextFusionProcessor.fuse().

    This is passed to Module 5 (RAG Ingestion) as the context document to
    be chunked, embedded, and stored. It is also the direct context input
    to Module 6 (LLM Legal Reasoning) for zero-shot legal queries.

    fused_text format::

        [OCR-DOCUMENT]
        Full text from the scanned PDF...
        [/OCR-DOCUMENT]

        [HTR-HANDWRITING]
        Handwritten annotations...
        [REVIEW]low-confidence region[/REVIEW]
        [/HTR-HANDWRITING]

        [ASR-VOICE]
        The plaintiff filed a case under Section 138...
        [/ASR-VOICE]

        [TYPED-INPUT]
        Draft a legal notice...
        [/TYPED-INPUT]
    """

    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID4 — links this fused doc to upstream modality results",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="document_id values from all contributing modality results",
    )
    modalities: list[ModalityBlock] = Field(
        default_factory=list,
        description="Per-modality blocks, in processing order (OCR→HTR→ASR→TYPED)",
    )
    fused_text: str = Field(
        "",
        description="Full tagged composite text — all modalities combined",
    )
    primary_language: str = Field(
        "en",
        description="Most-represented language across all modality inputs",
    )
    detected_intent: LegalIntent = Field(
        "unknown",
        description="High-level legal intent (from ASR or heuristic fallback)",
    )
    review_flags: list[ReviewFlag] = Field(
        default_factory=list,
        description="All [REVIEW] items from all modalities — for human QA",
    )
    total_word_count: int = Field(0, ge=0)
    processing_time_s: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def compute_total_words(self) -> "FusedDocument":
        if self.total_word_count == 0 and self.fused_text:
            self.total_word_count = len(self.fused_text.split())
        return self

    # ── Downstream contracts ───────────────────────────────────────────────

    def for_rag(self) -> str:
        """
        Plain-text view with all [SOURCE] / [/SOURCE] tags stripped.

        Ready for vector embedding — clean sentences, no XML noise.
        Preserves [REVIEW]…[/REVIEW] markers so the RAG layer can
        weight low-confidence spans lower during retrieval.
        """
        import re
        text = self.fused_text
        # Strip modality wrapper tags but keep REVIEW markers
        text = re.sub(r"\[(OCR-DOCUMENT|HTR-HANDWRITING|ASR-VOICE|TYPED-INPUT)\]\n?", "", text)
        text = re.sub(r"\[/(OCR-DOCUMENT|HTR-HANDWRITING|ASR-VOICE|TYPED-INPUT)\]\n?", "", text)
        return text.strip()

    def for_llm(self) -> str:
        """
        Full tagged text with a metadata header — ready for LLM context window.

        Includes language, intent, word count, and source summary so the
        LLM can calibrate its response (e.g. prefer OCR text for citations,
        use ASR intent to decide whether to answer or draft).
        """
        sources = ", ".join(b.source for b in self.modalities if b.text)
        header = (
            f"[LEGAL-AI CONTEXT]\n"
            f"Language: {self.primary_language}\n"
            f"Intent: {self.detected_intent}\n"
            f"Sources: {sources}\n"
            f"Word count: {self.total_word_count}\n"
            f"Review flags: {len(self.review_flags)}\n"
            f"[/LEGAL-AI CONTEXT]\n\n"
        )
        return header + self.fused_text

    def has_content(self) -> bool:
        """Return True if the fused document has any meaningful text."""
        return bool(self.fused_text.strip())

    def modality_text(self, source: ModalitySource) -> str:
        """Return the text contributed by a specific modality, or empty string."""
        for block in self.modalities:
            if block.source == source:
                return block.text
        return ""
