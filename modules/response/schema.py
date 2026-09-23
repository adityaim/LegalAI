"""
modules/response/schema.py
───────────────────────────
Pydantic v2 schema for Module 7: Response Generation & Formatting.

Contracts
─────────
• ResponseFormat  — output format enum (MARKDOWN, PLAIN, HTML, JSON)
• CitationBlock   — user-facing citation (index, source, excerpt, score)
• FormattedResponse — complete formatted answer delivered to UI/API

Design decisions
────────────────
• to_markdown() / to_plain() / to_html() are computed properties on the model,
  not stored fields, so the JSON schema stays clean.
• Citations are 1-indexed for human readability.
• to_html() generates minimal safe HTML (no external libs, no XSS risk from
  user content — all content is from our own pipeline).
"""

from __future__ import annotations

import html as _html
import re
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


# ─── Response format ──────────────────────────────────────────────────────────

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    PLAIN    = "plain"
    HTML     = "html"
    JSON     = "json"


# ─── Citation block ───────────────────────────────────────────────────────────

class CitationBlock(BaseModel):
    """User-facing citation entry."""

    index: int = Field(..., ge=1, description="1-based citation number")
    source_modality: str = Field("UNKNOWN")
    text_excerpt: str = Field(..., description="Chunk text excerpt (max 150 chars)")
    chunk_id: str = Field(...)
    relevance_score: float = Field(..., ge=0.0, le=1.0)

    def as_markdown(self) -> str:
        return (
            f"[{self.index}] **{self.source_modality}** "
            f"(score: {self.relevance_score:.3f}): "
            f"_{self.text_excerpt}_"
        )

    def as_plain(self) -> str:
        return (
            f"[{self.index}] {self.source_modality} "
            f"(score: {self.relevance_score:.3f}): "
            f"{self.text_excerpt}"
        )

    def as_html(self) -> str:
        excerpt = _html.escape(self.text_excerpt)
        modality = _html.escape(self.source_modality)
        return (
            f"<li><strong>[{self.index}]</strong> <em>{modality}</em> "
            f"(score: {self.relevance_score:.3f}): {excerpt}</li>"
        )


# ─── Formatted response ───────────────────────────────────────────────────────

class FormattedResponse(BaseModel):
    """
    Final user-facing response from the Legal AI pipeline.

    Produced by ResponseFormatter.format(legal_answer).
    """

    response_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID4 auto-generated per response",
    )
    query: str = Field(...)
    answer: str = Field(..., min_length=1, description="Formatted answer text")
    citations: list[CitationBlock] = Field(default_factory=list)
    format: ResponseFormat = Field(ResponseFormat.MARKDOWN)
    language: str = Field("en")
    confidence: float = Field(..., ge=0.0, le=100.0)
    is_grounded: bool = Field(False)
    review_required: bool = Field(False)
    processing_time_s: float = Field(..., ge=0.0)

    # ── Rendering methods ─────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """Render a complete Markdown document."""
        parts = [f"## Legal AI Answer\n\n{self.answer}"]
        if self.citations:
            parts.append("## Citations\n")
            parts.extend(c.as_markdown() for c in self.citations)
        meta = (
            f"\n---\n"
            f"*Confidence: {self.confidence:.1f}% | "
            f"Grounded: {'Yes' if self.is_grounded else 'No'} | "
            f"Language: {self.language}*"
        )
        parts.append(meta)
        return "\n\n".join(parts)

    def to_plain(self) -> str:
        """Render clean plain text."""
        parts = [f"ANSWER\n{'='*60}\n{self.answer}"]
        if self.citations:
            parts.append(f"\nCITATIONS\n{'-'*40}")
            parts.extend(c.as_plain() for c in self.citations)
        parts.append(
            f"\nConfidence: {self.confidence:.1f}% | "
            f"Grounded: {'Yes' if self.is_grounded else 'No'}"
        )
        return "\n".join(parts)

    def to_html(self) -> str:
        """Render minimal safe HTML."""
        answer_html = _html.escape(self.answer).replace("\n\n", "</p><p>").replace("\n", "<br>")
        parts = [
            "<html><body>",
            f"<h2>Legal AI Answer</h2><p>{answer_html}</p>",
        ]
        if self.citations:
            parts.append("<h3>Citations</h3><ol>")
            parts.extend(c.as_html() for c in self.citations)
            parts.append("</ol>")
        grounded_str = "Yes" if self.is_grounded else "No"
        parts.append(
            f"<p><em>Confidence: {self.confidence:.1f}% | "
            f"Grounded: {grounded_str}</em></p>"
        )
        parts.append("</body></html>")
        return "\n".join(parts)

    def to_json_summary(self) -> dict:
        """Compact dict for API inspection / health checks."""
        return {
            "response_id": self.response_id,
            "query": self.query,
            "answer_preview": self.answer[:200],
            "confidence": self.confidence,
            "citations": len(self.citations),
            "is_grounded": self.is_grounded,
            "review_required": self.review_required,
            "format": self.format.value,
        }
