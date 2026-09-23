"""
tests/response/test_schema.py
──────────────────────────────
Unit tests for Module 7 schema: ResponseFormat, CitationBlock, FormattedResponse.
"""

import pytest
from pydantic import ValidationError

from modules.response.schema import CitationBlock, FormattedResponse, ResponseFormat


# ─── ResponseFormat ───────────────────────────────────────────────────────────

class TestResponseFormat:
    def test_values(self):
        assert ResponseFormat.MARKDOWN == "markdown"
        assert ResponseFormat.PLAIN    == "plain"
        assert ResponseFormat.HTML     == "html"
        assert ResponseFormat.JSON     == "json"

    def test_str_equality(self):
        assert ResponseFormat("markdown") == ResponseFormat.MARKDOWN


# ─── CitationBlock ────────────────────────────────────────────────────────────

def make_citation(**kw) -> CitationBlock:
    defaults = dict(
        index=1,
        source_modality="OCR",
        text_excerpt="Section 138 of the NI Act deals with dishonour of cheque.",
        chunk_id="chunk-001",
        relevance_score=0.92,
    )
    defaults.update(kw)
    return CitationBlock(**defaults)


class TestCitationBlock:
    def test_valid(self):
        c = make_citation()
        assert c.index == 1
        assert c.relevance_score == 0.92

    def test_index_must_be_ge_1(self):
        with pytest.raises(ValidationError):
            make_citation(index=0)

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            make_citation(relevance_score=1.1)
        with pytest.raises(ValidationError):
            make_citation(relevance_score=-0.1)

    def test_as_markdown_contains_index(self):
        md = make_citation(index=3).as_markdown()
        assert "[3]" in md

    def test_as_markdown_contains_modality(self):
        md = make_citation(source_modality="ASR").as_markdown()
        assert "ASR" in md

    def test_as_markdown_contains_score(self):
        md = make_citation(relevance_score=0.876).as_markdown()
        assert "0.876" in md

    def test_as_plain_contains_index(self):
        pl = make_citation(index=2).as_plain()
        assert "[2]" in pl

    def test_as_plain_no_markdown(self):
        pl = make_citation().as_plain()
        assert "**" not in pl  # plain text, no bold

    def test_as_html_contains_li(self):
        h = make_citation().as_html()
        assert "<li>" in h
        assert "</li>" in h

    def test_serialisation(self):
        c = make_citation()
        data = c.model_dump()
        restored = CitationBlock.model_validate(data)
        assert restored.index == 1


# ─── FormattedResponse ────────────────────────────────────────────────────────

def make_response(**kw) -> FormattedResponse:
    defaults = dict(
        query="What is Section 138?",
        answer="Section 138 NI Act deals with dishonour of cheque.",
        citations=[make_citation()],
        format=ResponseFormat.MARKDOWN,
        language="en",
        confidence=85.0,
        is_grounded=True,
        review_required=False,
        processing_time_s=0.02,
    )
    defaults.update(kw)
    return FormattedResponse(**defaults)


class TestFormattedResponse:
    def test_auto_response_id(self):
        r1 = make_response()
        r2 = make_response()
        assert r1.response_id != r2.response_id
        assert len(r1.response_id) == 36

    def test_to_markdown_has_header(self):
        md = make_response().to_markdown()
        assert "## Legal AI Answer" in md

    def test_to_markdown_has_citations_section(self):
        md = make_response().to_markdown()
        assert "## Citations" in md

    def test_to_markdown_no_citations_section_when_empty(self):
        md = make_response(citations=[]).to_markdown()
        assert "## Citations" not in md

    def test_to_markdown_contains_answer_text(self):
        md = make_response().to_markdown()
        assert "dishonour of cheque" in md

    def test_to_plain_no_markdown_syntax(self):
        pl = make_response().to_plain()
        assert "##" not in pl
        assert "**" not in pl

    def test_to_plain_has_answer_header(self):
        pl = make_response().to_plain()
        assert "ANSWER" in pl

    def test_to_html_has_html_tags(self):
        h = make_response().to_html()
        assert "<html>" in h
        assert "</html>" in h
        assert "<h2>" in h

    def test_to_html_citations_in_ol(self):
        h = make_response().to_html()
        assert "<ol>" in h

    def test_to_json_summary_keys(self):
        s = make_response().to_json_summary()
        for key in ("response_id", "query", "answer_preview", "confidence", "citations", "is_grounded"):
            assert key in s

    def test_to_json_summary_answer_preview_truncated(self):
        long_answer = "word " * 200
        s = make_response(answer=long_answer).to_json_summary()
        assert len(s["answer_preview"]) <= 200

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            make_response(confidence=101.0)
        with pytest.raises(ValidationError):
            make_response(confidence=-1.0)

    def test_serialisation_round_trip(self):
        r = make_response()
        data = r.model_dump()
        restored = FormattedResponse.model_validate(data)
        assert restored.response_id == r.response_id
        assert restored.confidence == r.confidence
