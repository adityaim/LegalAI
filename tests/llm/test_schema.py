"""
tests/llm/test_schema.py
─────────────────────────
Unit tests for Module 6 schema: Citation, LegalQuery, LegalAnswer.
"""

import pytest
from pydantic import ValidationError

from modules.llm.schema import Citation, LegalAnswer, LegalQuery


# ─── Citation ─────────────────────────────────────────────────────────────────

class TestCitation:
    def make(self, **kw) -> Citation:
        defaults = dict(
            chunk_id="chunk-001",
            document_id="doc-001",
            source_modality="OCR",
            text_excerpt="Section 138 of the NI Act.",
            relevance_score=0.92,
        )
        defaults.update(kw)
        return Citation(**defaults)

    def test_valid(self):
        c = self.make()
        assert c.chunk_id == "chunk-001"
        assert c.relevance_score == 0.92

    def test_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            self.make(relevance_score=-0.1)

    def test_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            self.make(relevance_score=1.1)

    def test_default_modality(self):
        c = Citation(
            chunk_id="x", document_id="y",
            text_excerpt="text", relevance_score=0.5,
        )
        assert c.source_modality == "UNKNOWN"


# ─── LegalQuery ───────────────────────────────────────────────────────────────

class TestLegalQuery:
    def test_valid_minimal(self):
        q = LegalQuery(query="What is Section 138?")
        assert q.query == "What is Section 138?"
        assert q.language == "en"
        assert q.fused_document_text == ""
        assert q.retrieval_context == ""

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            LegalQuery(query="")

    def test_max_tokens_bounds(self):
        with pytest.raises(ValidationError):
            LegalQuery(query="test", max_tokens=10)
        with pytest.raises(ValidationError):
            LegalQuery(query="test", max_tokens=9999)

    def test_with_context(self):
        q = LegalQuery(
            query="test",
            fused_document_text="[OCR-DOCUMENT]...[/OCR-DOCUMENT]",
            retrieval_context="[CONTEXT]...[/CONTEXT]",
        )
        assert "[OCR-DOCUMENT]" in q.fused_document_text
        assert "[CONTEXT]" in q.retrieval_context


# ─── LegalAnswer ──────────────────────────────────────────────────────────────

def make_answer(**kw) -> LegalAnswer:
    defaults = dict(
        query="What is Section 138?",
        answer_text="Section 138 NI Act deals with dishonour of cheque [1].",
        citations=[Citation(
            chunk_id="c1", document_id="d1",
            source_modality="OCR",
            text_excerpt="Section 138...",
            relevance_score=0.92,
        )],
        confidence=85.0,
        language="en",
        detected_intent="question",
        processing_time_s=0.05,
        llm_engine="mock",
        is_grounded=True,
    )
    defaults.update(kw)
    return LegalAnswer(**defaults)


class TestLegalAnswer:
    def test_auto_answer_id(self):
        a1 = make_answer()
        a2 = make_answer()
        assert a1.answer_id != a2.answer_id
        assert len(a1.answer_id) == 36   # UUID4

    def test_is_grounded_true_with_citations(self):
        a = make_answer()
        assert a.is_grounded is True

    def test_is_grounded_false_no_citations(self):
        a = make_answer(citations=[])
        assert a.is_grounded is False

    def test_review_required_low_confidence(self):
        a = make_answer(confidence=50.0)
        assert a.review_required is True

    def test_review_required_no_grounding(self):
        a = make_answer(citations=[], confidence=85.0)
        assert a.review_required is True

    def test_review_required_false(self):
        a = make_answer(confidence=85.0)
        assert a.review_required is False

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            make_answer(confidence=101.0)
        with pytest.raises(ValidationError):
            make_answer(confidence=-1.0)

    def test_empty_answer_text_rejected(self):
        with pytest.raises(ValidationError):
            make_answer(answer_text="")

    def test_processing_time_non_negative(self):
        with pytest.raises(ValidationError):
            make_answer(processing_time_s=-0.1)

    def test_summary_dict(self):
        a = make_answer()
        s = a.summary()
        assert "answer_id" in s
        assert "confidence" in s
        assert s["citations"] == 1
        assert s["is_grounded"] is True

    def test_serialisation_round_trip(self):
        a = make_answer()
        data = a.model_dump()
        restored = LegalAnswer.model_validate(data)
        assert restored.answer_id == a.answer_id
        assert restored.confidence == a.confidence
        assert len(restored.citations) == 1

    def test_detected_intent_preserved(self):
        a = make_answer(detected_intent="instruction")
        assert a.detected_intent == "instruction"
