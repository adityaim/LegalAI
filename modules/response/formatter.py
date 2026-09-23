"""
modules/response/formatter.py
───────────────────────────────
ResponseFormatter: converts a LegalAnswer (Module 6) into a FormattedResponse (Module 7).

Pipeline:
    LegalAnswer
        -> strip internal markers ([REVIEW]...[/REVIEW], [UNCERTAIN])
        -> CitationBuilder.build() -> list[CitationBlock]
        -> FormattedResponse

Internal markers stripped
─────────────────────────
• [REVIEW]text[/REVIEW]  -> text  (keeps inner content)
• [UNCERTAIN]            -> ""    (removes the tag entirely)
These markers are pipeline-internal signals; end users should not see them.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .citation_builder import CitationBuilder
from .schema import CitationBlock, FormattedResponse, ResponseFormat

# ── Regex patterns ────────────────────────────────────────────────────────────

_REVIEW_RE    = re.compile(r"\[REVIEW\](.*?)\[/REVIEW\]", re.DOTALL)
_UNCERTAIN_RE = re.compile(r"\[UNCERTAIN\]")


@dataclass
class FormatterConfig:
    """Configuration for ResponseFormatter."""
    default_format: ResponseFormat = ResponseFormat.MARKDOWN
    strip_internal_markers: bool = True   # remove [REVIEW]/[UNCERTAIN] wrappers
    max_citations: int = 10
    include_confidence: bool = True


class ResponseFormatter:
    """
    Formats a LegalAnswer into a user-facing FormattedResponse.

    Usage::

        formatter = ResponseFormatter()
        response = formatter.format(legal_answer)
        print(response.to_markdown())
    """

    def __init__(self, config: FormatterConfig | None = None) -> None:
        self.config = config or FormatterConfig()
        self._citation_builder = CitationBuilder()

    def format(
        self,
        legal_answer,             # LegalAnswer (duck-typed to avoid circular import in tests)
        fmt: ResponseFormat | str | None = None,
    ) -> FormattedResponse:
        """
        Convert a LegalAnswer into a FormattedResponse.

        Parameters
        ----------
        legal_answer : LegalAnswer
            Output of LegalReasoner.reason().
        fmt : ResponseFormat | str | None
            Override the default output format.

        Returns
        -------
        FormattedResponse
        """
        t0 = time.perf_counter()

        # Resolve format
        if fmt is None:
            output_format = self.config.default_format
        elif isinstance(fmt, str):
            output_format = ResponseFormat(fmt.lower())
        else:
            output_format = fmt

        # Strip internal markers
        answer_text = legal_answer.answer_text
        if self.config.strip_internal_markers:
            answer_text = self._strip_markers(answer_text)

        # Build citation blocks
        citation_blocks: list[CitationBlock] = self._citation_builder.build(
            citations=legal_answer.citations,
            max_citations=self.config.max_citations,
        )

        elapsed = time.perf_counter() - t0

        return FormattedResponse(
            query=legal_answer.query,
            answer=answer_text.strip(),
            citations=citation_blocks,
            format=output_format,
            language=legal_answer.language,
            confidence=legal_answer.confidence,
            is_grounded=legal_answer.is_grounded,
            review_required=legal_answer.review_required,
            processing_time_s=round(elapsed, 4),
        )

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _strip_markers(text: str) -> str:
        """
        Remove pipeline-internal markers from answer text.

        [REVIEW]uncertain text[/REVIEW] -> uncertain text
        [UNCERTAIN]                      -> (removed)
        """
        text = _REVIEW_RE.sub(r"\1", text)
        text = _UNCERTAIN_RE.sub("", text)
        # Collapse any double spaces left by removed tags
        text = re.sub(r" {2,}", " ", text)
        return text
