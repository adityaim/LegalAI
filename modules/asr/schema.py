"""
modules/asr/schema.py
──────────────────────
Pydantic v2 output schema for Module 3: ASR Voice-to-Text.

Mirrors the OCR/HTR schema structure for consistency — all modules feed
into the same Text Fusion layer (Module 4).

Design spec (Section 4.3):
  - ASRSegment per time-aligned speech segment (≈ sentence)
  - confidence from Whisper avg_logprob → 0-100 mapping
  - review_required when confidence < 70
  - intent: detected legal intent class (question / instruction / narration)
  - to_fusion_text() wraps low-confidence segments in [REVIEW]…[/REVIEW]

Reference:
  SYSTRAN/faster-whisper — Segment dataclass (start, end, text, avg_logprob,
  no_speech_prob, words) used to populate ASRSegment fields.
  https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

# ─── Type aliases ─────────────────────────────────────────────────────────────

ASREngineTag = Literal["whisper", "faster_whisper", "mock", "unknown"]

LegalIntent = Literal[
    "question",       # "What does Section 138 say about..."
    "instruction",    # "Draft a notice under Section 420..."
    "narration",      # "The accused appeared before the court on..."
    "unknown",
]


# ─── Segment model ────────────────────────────────────────────────────────────

class ASRSegment(BaseModel):
    """
    A single time-aligned speech segment (roughly a sentence).

    Analogous to OCRBlock (Module 1) and HTRRegion (Module 2).

    Confidence mapping from Whisper avg_logprob:
      avg_logprob is in (-∞, 0]; we map it to 0–100 via:
        confidence = max(0, min(100, (1 + avg_logprob / 4) * 100))
      This makes logprob=0 → 100%, logprob=-4 → 0%, matching the
      design-spec 70% review threshold in practical usage.
    """

    segment_id: int = Field(..., ge=0, description="0-indexed segment number")
    start_s: float = Field(..., ge=0.0, description="Segment start time in seconds")
    end_s: float = Field(..., ge=0.0, description="Segment end time in seconds")
    text: str = Field(..., description="Recognised UTF-8 text for this segment")
    confidence: float = Field(
        ..., ge=0.0, le=100.0,
        description=(
            "Confidence score 0–100 derived from Whisper avg_logprob. "
            "Formula: max(0, min(100, (1 + avg_logprob / 4) * 100))"
        ),
    )
    no_speech_prob: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Probability that this segment contains no speech (from Whisper)",
    )
    review_required: bool = Field(
        False,
        description="True when confidence < 70 — highlight in UI for human verification",
    )
    asr_engine: ASREngineTag = Field(
        "unknown",
        description="Which ASR engine produced this segment's text",
    )
    language: str = Field(
        "en",
        description="ISO 639-1 language code for this segment (from Whisper detection)",
    )

    @model_validator(mode="after")
    def auto_review_required(self) -> "ASRSegment":
        """Auto-set review_required when confidence < 70. Explicit True is preserved."""
        if not self.review_required and self.confidence < 70.0:
            self.review_required = True
        return self

    @model_validator(mode="after")
    def end_after_start(self) -> "ASRSegment":
        """Ensure end_s >= start_s."""
        if self.end_s < self.start_s:
            raise ValueError(
                f"end_s ({self.end_s}) must be >= start_s ({self.start_s})"
            )
        return self


# ─── Top-level result ─────────────────────────────────────────────────────────

class ASRResult(BaseModel):
    """
    Top-level ASR output returned by ASRReader.process().

    This is the canonical output contract between Module 3 (ASR) and all
    downstream modules (Module 4 Text Fusion, Module 5 RAG Ingestion).
    """

    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID4 assigned at ingestion time",
    )
    source: Literal["ASR"] = "ASR"
    language: str = Field(
        "en",
        description="Primary language detected by Whisper (ISO 639-1)",
    )
    language_probability: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Whisper's confidence in the detected language (0–1)",
    )
    duration_s: float = Field(
        ..., ge=0.0,
        description="Total audio duration in seconds",
    )
    transcript: str = Field(
        "",
        description="Full transcript — all segments joined in order",
    )
    segments: list[ASRSegment] = Field(
        default_factory=list,
        description="All time-aligned segments, in chronological order",
    )
    intent: LegalIntent = Field(
        "unknown",
        description="Detected high-level legal intent of the utterance",
    )
    processing_time_s: float = Field(
        ..., ge=0.0,
        description="Wall-clock processing time in seconds",
    )
    low_confidence_segments: int = Field(
        0, ge=0,
        description="Count of segments with review_required=True",
    )

    # ── Helpers ────────────────────────────────────────────────────────────

    def mean_confidence(self) -> float:
        """Overall mean confidence across all segments (0–100)."""
        if not self.segments:
            return 0.0
        return sum(s.confidence for s in self.segments) / len(self.segments)

    def flagged_segments(self) -> list[ASRSegment]:
        """Return only segments that require human review."""
        return [s for s in self.segments if s.review_required]

    def to_fusion_text(self) -> str:
        """
        Format transcript for Module 4 Text Fusion.

        Low-confidence segments are wrapped in [REVIEW]…[/REVIEW] markers
        so the fusion layer can flag them for downstream handling.
        Mirrors the pattern from OCRResult and HTRResult.
        """
        parts: list[str] = []
        for seg in self.segments:
            text = seg.text.strip()
            if not text:
                continue
            if seg.review_required:
                parts.append(f"[REVIEW]{text}[/REVIEW]")
            else:
                parts.append(text)
        return "\n".join(parts)
