"""
modules/htr/schema.py
──────────────────────
Pydantic v2 output schema for Module 2: Handwritten Document Reader (HTR).

Mirrors the OCR schema structure for consistency — both modules output to
the same Text Fusion layer (Module 4). BoundingBox is imported from the
OCR module to avoid duplication.

Design spec (Section 4.2):
  - HTRRegion per detected handwritten region
  - confidence from TrOCR token log-probabilities
  - review_required when confidence < 70
  - type: standalone_note | annotation | signature | table_fill | unknown
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

# Reuse the canonical BoundingBox from Module 1
from modules.ocr.schema import BoundingBox


# ─── Type aliases ─────────────────────────────────────────────────────────────

HandwritingType = Literal[
    "standalone_note",   # a self-contained handwritten note/paragraph
    "annotation",        # overlay on printed text (margin note, underline comment)
    "signature",         # signature or initials block
    "table_fill",        # handwritten value inside a printed table/form cell
    "unknown",
]

HTREngineTag = Literal["trocr", "tesseract_devanagari", "mock", "unknown"]


# ─── Region model ─────────────────────────────────────────────────────────────

class HTRRegion(BaseModel):
    """
    A single recognised handwritten region.

    Analogous to OCRBlock in Module 1 but specific to handwriting:
    includes handwriting type classification and per-token confidence.
    """

    page: int = Field(..., ge=1, description="1-indexed page number")
    type: HandwritingType = Field(..., description="Classified handwriting region type")
    text: str = Field(..., description="Recognised UTF-8 text for this region")
    confidence: float = Field(
        ..., ge=0.0, le=100.0,
        description="Confidence score 0–100 (from TrOCR token log-probabilities)"
    )
    bbox: BoundingBox = Field(..., description="Bounding box in page-pixel coords")
    review_required: bool = Field(
        False,
        description="True when confidence < 70 — highlight in UI for human verification",
    )
    htr_engine: HTREngineTag = Field(
        "unknown",
        description="Which HTR engine produced this region's text",
    )
    language: str = Field(
        "en",
        description="ISO 639-1 language code for the recognised text",
    )

    @model_validator(mode="after")
    def auto_review_required(self) -> "HTRRegion":
        """Auto-set review_required when confidence < 70. Explicit True is preserved."""
        if not self.review_required and self.confidence < 70.0:
            self.review_required = True
        return self


# ─── Top-level result ─────────────────────────────────────────────────────────

class HTRResult(BaseModel):
    """
    Top-level HTR output returned by HandwritingReader.process().

    This is the canonical output contract between Module 2 (HTR) and all
    downstream modules (Module 4 Text Fusion, Module 5 RAG Ingestion).
    """

    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID4 assigned at ingestion time",
    )
    source: Literal["HTR"] = "HTR"
    input_type: str = Field(
        "image",
        description="'image' for standalone image files, 'pdf_page' for extracted PDF pages",
    )
    language: str = Field("en", description="Primary language detected in the handwriting")
    regions: list[HTRRegion] = Field(
        default_factory=list,
        description="All detected handwritten regions, in reading order",
    )
    full_text: str = Field(
        "",
        description="Concatenated text from all regions (review_required regions marked)",
    )
    processing_time_s: float = Field(..., ge=0.0, description="Wall-clock time in seconds")
    low_confidence_regions: int = Field(
        0, ge=0,
        description="Count of regions with review_required=True",
    )

    # ── Helpers ────────────────────────────────────────────────────────────

    def mean_confidence(self) -> float:
        """Overall mean confidence across all regions (0–100)."""
        if not self.regions:
            return 0.0
        return sum(r.confidence for r in self.regions) / len(self.regions)

    def flagged_regions(self) -> list[HTRRegion]:
        """Return only regions that require human review."""
        return [r for r in self.regions if r.review_required]

    def regions_by_type(self, hw_type: HandwritingType) -> list[HTRRegion]:
        """Return regions matching the given handwriting type."""
        return [r for r in self.regions if r.type == hw_type]

    def to_fusion_text(self) -> str:
        """
        Format full_text for Module 4 Text Fusion.

        Low-confidence regions are wrapped in [REVIEW] ... [/REVIEW] markers
        so the fusion layer can flag them for downstream handling.
        """
        parts: list[str] = []
        for region in self.regions:
            text = region.text.strip()
            if not text:
                continue
            if region.review_required:
                parts.append(f"[REVIEW]{text}[/REVIEW]")
            else:
                parts.append(text)
        return "\n".join(parts)
