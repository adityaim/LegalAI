"""
modules/ocr/schema.py
─────────────────────
Pydantic v2 output schema for the OCR Document Reader (Module 1).

Exact schema from Section 4.1 of the system design document, extended
with bounding boxes, per-block OCR engine tag, and processing metadata.
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Primitive models ────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Page-relative bounding box in pixels (origin: top-left)."""

    x0: float = Field(..., ge=0, description="Left edge (px)")
    y0: float = Field(..., ge=0, description="Top edge (px)")
    x1: float = Field(..., ge=0, description="Right edge (px)")
    y1: float = Field(..., ge=0, description="Bottom edge (px)")
    page: int = Field(..., ge=1, description="1-indexed page number")

    @field_validator("x1")
    @classmethod
    def x1_gt_x0(cls, v: float, info) -> float:
        if "x0" in info.data and v <= info.data["x0"]:
            raise ValueError("x1 must be greater than x0")
        return v

    @field_validator("y1")
    @classmethod
    def y1_gt_y0(cls, v: float, info) -> float:
        if "y0" in info.data and v <= info.data["y0"]:
            raise ValueError("y1 must be greater than y0")
        return v

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height


# ─── Block types ─────────────────────────────────────────────────────────────

BlockType = Literal["heading", "body", "table", "footer", "page_number", "unknown"]

OCREngineTag = Literal["pymupdf_text", "paddle", "tesseract", "unknown"]


# ─── Block model ─────────────────────────────────────────────────────────────

class OCRBlock(BaseModel):
    """
    A single detected layout block within a page.

    `review_required` is set True when confidence < LOW_CONFIDENCE_THRESHOLD
    (70.0) — the UI must highlight these for human verification.
    """

    page: int = Field(..., ge=1, description="1-indexed page number")
    type: BlockType = Field(..., description="Semantic block type")
    text: str = Field(..., description="Extracted / recognised text for this block")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Mean word-level confidence (0–100)",
    )
    bbox: BoundingBox | None = Field(
        None, description="Bounding box in page-pixel coords; None for native text"
    )
    review_required: bool = Field(
        False,
        description="True when confidence < 70 — flag for human review",
    )
    ocr_engine: OCREngineTag = Field(
        "unknown",
        description="Which engine produced this block's text",
    )

    @model_validator(mode="after")
    def auto_review_required(self) -> "OCRBlock":
        """
        Auto-derive review_required from confidence after all fields are set.
        Callers may still explicitly pass review_required=True to force flagging.
        This runs after field validation so confidence is always available.
        """
        # Only override if caller passed False (the default) — respect explicit True
        if not self.review_required and self.confidence < 70.0:
            self.review_required = True
        return self


# ─── Metadata model ──────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """
    Lightweight document-level metadata extracted during post-processing.
    Populated by heuristic pattern matching; not guaranteed to be complete.
    """

    doc_type: str | None = Field(
        None,
        description="Detected document type (contract, FIR, court_order, …)",
    )
    parties: list[str] = Field(
        default_factory=list,
        description="Named parties extracted from the document",
    )
    date: str | None = Field(
        None,
        description="Primary document date (ISO 8601 where parseable)",
    )
    jurisdiction: str | None = Field(
        None,
        description="Detected jurisdiction (state / court / city)",
    )


# ─── Top-level result ────────────────────────────────────────────────────────

class OCRResult(BaseModel):
    """
    Top-level OCR output object returned by OCRDocumentReader.process().

    This is the canonical contract between Module 1 (OCR) and all downstream
    modules (Text Fusion, RAG, etc.).
    """

    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID v4 assigned at ingestion time",
    )
    source: Literal["OCR"] = "OCR"
    language: str = Field(
        "en",
        description="ISO 639-1 language code of the dominant language detected",
    )
    pages: int = Field(..., ge=1, description="Total number of pages processed")
    full_text: str = Field(
        ..., description="Concatenated clean text from all blocks, in reading order"
    )
    blocks: list[OCRBlock] = Field(
        default_factory=list,
        description="Per-block results in reading order",
    )
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    processing_time_s: float = Field(
        ..., ge=0.0, description="Wall-clock processing time in seconds"
    )
    low_confidence_pages: list[int] = Field(
        default_factory=list,
        description="1-indexed page numbers where mean confidence < 70",
    )

    # ── Helpers ────────────────────────────────────────────────────────────

    def pages_needing_review(self) -> list[int]:
        """Return 1-indexed page numbers that contain at least one review block."""
        return sorted(
            {b.page for b in self.blocks if b.review_required}
        )

    def mean_confidence(self) -> float:
        """Overall mean confidence across all blocks (0–100)."""
        if not self.blocks:
            return 0.0
        return sum(b.confidence for b in self.blocks) / len(self.blocks)

    def text_blocks(self) -> list[OCRBlock]:
        """Return only body + heading blocks (excludes footers / page numbers)."""
        return [b for b in self.blocks if b.type in ("body", "heading")]
