"""
tests/response/test_formatter.py
──────────────────────────────────
Unit tests for Module 7 formatter and citation builder.
All tests use hardcoded LegalAnswer — no real LLM calls.
"""

import pytest

from modules.llm.schema import Citation, LegalAnswer
from modules.response import ResponseFormatter
from modules.response.citation_builder import CitationBuilder
from modules.response.formatter import FormatterConfig
from modules.response.schema import FormattedResponse, ResponseFormat


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_citation(chunk_id="c1", score=0.92, modality="OCR",
                  text="Section 138 NI Act.") -> Citation:
    return Citation(
        chunk_id=chunk_id,
        document_id="doc-001",
        source_modality=modality,
        text_excerpt=text,
        relevance_score=score,
    )


def make_answer(**kw) -> LegalAnswer:
    defaults = dict(
        query="What is Section 138?",
        answer_text="Section 138 NI Act [1] deals with dishonour. Punishment [2] is two years.",
        citations=[make_citation("c1", 0.92, "OCR"), make_citation("c2", 0.80, "ASR", "ASR text.")],
        confidence=85.0,
        language="en",
        detected_intent="question",
        processing_time_s=0.5,
        llm_engine="mock",
        is_grounded=True,
    )
    defaults.update(kw)
    return LegalAnswer(**defaults)


# ─── CitationBuilder ──────────────────────────────────────────────────────────

class TestCitationBuilder:
    def test_build_returns_citation_blocks(self):
        builder = CitationBuilder()
        cits = [make_citation("c1", 0.9), make_citation("c2", 0.7)]
        blocks = builder.build(cits)
        assert len(blocks) == 2

    def test_sorted_by_score_descending(self):
        builder = CitationBuilder()
        cits = [make_citation("c1", 0.5), make_citation("c2", 0.9)]
        blocks = builder.build(cits)
        assert blocks[0].relevance_score >= blocks[1].relevance_score

    def test_indices_are_1_based(self):
        builder = CitationBuilder()
        cits = [make_citation("c1"), make_citation("c2")]
        blocks = builder.build(cits)
        for i, b in enumerate(blocks, 1):
            assert b.index == i

    def test_deduplication_by_chunk_id(self):
        builder = CitationBuilder()
        cits = [make_citation("c1", 0.9), make_citation("c1", 0.7)]  # same chunk_id
        blocks = builder.build(cits)
        assert len(blocks) == 1
        assert blocks[0].relevance_score == 0.9  # kept highest

    def test_empty_input(self):
        builder = CitationBuilder()
        assert builder.build([]) == []

    def test_max_citations_respected(self):
        builder = CitationBuilder()
        cits = [make_citation(f"c{i}", 0.9 - i * 0.05) for i in range(15)]
        blocks = builder.build(cits, max_citations=5)
        assert len(blocks) == 5

    def test_text_excerpt_truncated(self):
        long_text = "word " * 100  # 500 chars
        builder = CitationBuilder()
        blocks = builder.build([make_citation("c1", text=long_text)])
        assert len(blocks[0].text_excerpt) <= 153  # 150 + "..."

    def test_modality_preserved(self):
        builder = CitationBuilder()
        blocks = builder.build([make_citation("c1", modality="HTR")])
        assert blocks[0].source_modality == "HTR"


# ─── ResponseFormatter ────────────────────────────────────────────────────────

class TestResponseFormatter:
    def test_format_returns_formatted_response(self):
        f = ResponseFormatter()
        r = f.format(make_answer())
        assert isinstance(r, FormattedResponse)

    def test_default_format_is_markdown(self):
        f = ResponseFormatter()
        r = f.format(make_answer())
        assert r.format == ResponseFormat.MARKDOWN

    def test_format_plain(self):
        f = ResponseFormatter()
        r = f.format(make_answer(), fmt="plain")
        assert r.format == ResponseFormat.PLAIN

    def test_format_html(self):
        f = ResponseFormatter()
        r = f.format(make_answer(), fmt=ResponseFormat.HTML)
        assert r.format == ResponseFormat.HTML

    def test_format_json(self):
        f = ResponseFormatter()
        r = f.format(make_answer(), fmt=ResponseFormat.JSON)
        assert r.format == ResponseFormat.JSON

    def test_strips_review_markers(self):
        ans = make_answer(answer_text="Normal text. [REVIEW]flagged part[/REVIEW] More text.")
        f = ResponseFormatter()
        r = f.format(ans)
        assert "[REVIEW]" not in r.answer
        assert "[/REVIEW]" not in r.answer
        assert "flagged part" in r.answer  # inner content preserved

    def test_strips_uncertain_tags(self):
        ans = make_answer(answer_text="Some [UNCERTAIN] statement about law.")
        f = ResponseFormatter()
        r = f.format(ans)
        assert "[UNCERTAIN]" not in r.answer

    def test_no_strip_when_disabled(self):
        cfg = FormatterConfig(strip_internal_markers=False)
        f = ResponseFormatter(config=cfg)
        ans = make_answer(answer_text="[REVIEW]text[/REVIEW]")
        r = f.format(ans)
        assert "[REVIEW]" in r.answer

    def test_citations_built(self):
        f = ResponseFormatter()
        r = f.format(make_answer())
        assert len(r.citations) == 2

    def test_no_citations_when_empty(self):
        f = ResponseFormatter()
        r = f.format(make_answer(citations=[]))
        assert r.citations == []

    def test_confidence_propagated(self):
        f = ResponseFormatter()
        r = f.format(make_answer(confidence=72.5))
        assert r.confidence == 72.5

    def test_is_grounded_propagated(self):
        f = ResponseFormatter()
        r = f.format(make_answer())
        assert r.is_grounded is True

    def test_review_required_propagated(self):
        f = ResponseFormatter()
        r = f.format(make_answer(citations=[], confidence=50.0))
        assert r.review_required is True

    def test_language_propagated(self):
        f = ResponseFormatter()
        r = f.format(make_answer(language="hi"))
        assert r.language == "hi"

    def test_processing_time_non_negative(self):
        f = ResponseFormatter()
        r = f.format(make_answer())
        assert r.processing_time_s >= 0.0

    def test_query_propagated(self):
        f = ResponseFormatter()
        r = f.format(make_answer())
        assert r.query == "What is Section 138?"

    def test_max_citations_config(self):
        cfg = FormatterConfig(max_citations=1)
        f = ResponseFormatter(config=cfg)
        r = f.format(make_answer())
        assert len(r.citations) == 1

    def test_serialisation_round_trip(self):
        f = ResponseFormatter()
        r = f.format(make_answer())
        data = r.model_dump()
        restored = FormattedResponse.model_validate(data)
        assert restored.response_id == r.response_id
        assert restored.confidence == r.confidence
