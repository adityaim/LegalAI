"""
modules/response/citation_builder.py
──────────────────────────────────────
CitationBuilder: converts Citation objects (from LegalAnswer) into
sorted, deduplicated, 1-indexed CitationBlock objects.
"""

from __future__ import annotations

from .schema import CitationBlock

# Maximum excerpt length (chars)
_MAX_EXCERPT = 150


class CitationBuilder:
    """
    Converts a list of Citation objects from a LegalAnswer into user-facing
    CitationBlock objects.

    Rules:
    1. Deduplicate by chunk_id — keep the entry with the highest relevance_score.
    2. Sort descending by relevance_score.
    3. Assign 1-based sequential index numbers.
    4. Truncate text_excerpt to _MAX_EXCERPT characters.
    """

    def build(self, citations: list, max_citations: int = 10) -> list[CitationBlock]:
        """
        Parameters
        ----------
        citations : list[Citation]
            Citation objects from LegalAnswer.citations.
        max_citations : int
            Maximum number of CitationBlock objects to return.

        Returns
        -------
        list[CitationBlock]
            1-indexed, sorted by score descending, deduplicated.
        """
        if not citations:
            return []

        # Deduplicate by chunk_id, keep highest score
        best: dict[str, object] = {}
        for cit in citations:
            existing = best.get(cit.chunk_id)
            if existing is None or cit.relevance_score > existing.relevance_score:
                best[cit.chunk_id] = cit

        # Sort by score descending
        sorted_cits = sorted(best.values(), key=lambda c: c.relevance_score, reverse=True)

        # Limit
        sorted_cits = sorted_cits[:max_citations]

        # Build CitationBlocks
        blocks: list[CitationBlock] = []
        for i, cit in enumerate(sorted_cits, 1):
            excerpt = cit.text_excerpt[:_MAX_EXCERPT]
            if len(cit.text_excerpt) > _MAX_EXCERPT:
                excerpt = excerpt.rstrip() + "..."
            blocks.append(CitationBlock(
                index=i,
                source_modality=cit.source_modality,
                text_excerpt=excerpt,
                chunk_id=cit.chunk_id,
                relevance_score=cit.relevance_score,
            ))

        return blocks
