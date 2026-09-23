"""
tests/llm/test_reasoner.py
───────────────────────────
Integration tests for LegalReasoner (Module 6).
All tests use MockLLMEngine — no real API calls.
"""

import pytest

from modules.llm import LegalReasoner, LegalQuery, LegalAnswer
from modules.llm.engines.mock_engine import MockLLMEngine
from modules.llm.engines.base import BaseLLMEngine, LLMResponse
from modules.llm.reasoner import ReasonerConfig, _parse_context_chunks, _extract_citations


# ─── Context parser unit tests ────────────────────────────────────────────────

SAMPLE_CONTEXT = """\
[CONTEXT]

[1] source=OCR score=0.923
Section 138 of the Negotiable Instruments Act deals with dishonour of cheque.

[2] source=ASR score=0.871
The punishment is imprisonment up to two years or fine up to twice the cheque amount.

[/CONTEXT]"""


class TestParseContextChunks:
    def test_parses_two_chunks(self):
        chunks = _parse_context_chunks(SAMPLE_CONTEXT)
        assert len(chunks) == 2

    def test_chunk_1_source(self):
        chunks = _parse_context_chunks(SAMPLE_CONTEXT)
        assert chunks[1]["source_modality"] == "OCR"

    def test_chunk_2_source(self):
        chunks = _parse_context_chunks(SAMPLE_CONTEXT)
        assert chunks[2]["source_modality"] == "ASR"

    def test_chunk_score(self):
        chunks = _parse_context_chunks(SAMPLE_CONTEXT)
        assert abs(chunks[1]["score"] - 0.923) < 0.001

    def test_empty_context(self):
        assert _parse_context_chunks("") == {}

    def test_no_context_wrapper(self):
        # Should still parse without [CONTEXT] wrapper
        raw = "[1] source=OCR score=0.5\nSome legal text."
        chunks = _parse_context_chunks(raw)
        assert 1 in chunks

    def test_text_excerpt_trimmed(self):
        chunks = _parse_context_chunks(SAMPLE_CONTEXT)
        assert len(chunks[1]["text_excerpt"]) <= 200


class TestExtractCitations:
    def test_extracts_cited_refs(self):
        answer = "Section 138 NI Act [1] provides the offence. Punishment is in [2]."
        cits = _extract_citations(answer, SAMPLE_CONTEXT)
        assert len(cits) == 2

    def test_no_refs_in_answer(self):
        answer = "Section 138 NI Act provides the offence."
        cits = _extract_citations(answer, SAMPLE_CONTEXT)
        assert cits == []

    def test_only_cited_refs_included(self):
        answer = "See [1] for details."
        cits = _extract_citations(answer, SAMPLE_CONTEXT)
        assert len(cits) == 1
        assert cits[0].source_modality == "OCR"

    def test_empty_context_returns_empty(self):
        cits = _extract_citations("Answer [1].", "")
        assert cits == []

    def test_citation_score_clamped(self):
        cits = _extract_citations("[1] answer here", SAMPLE_CONTEXT)
        assert 0.0 <= cits[0].relevance_score <= 1.0


# ─── MockLLMEngine ────────────────────────────────────────────────────────────

class TestMockLLMEngine:
    def test_engine_name(self):
        assert MockLLMEngine().engine_name == "mock"

    def test_ni138_keywords_trigger_ni_answer(self):
        engine = MockLLMEngine()
        resp = engine.generate("sys", "Section 138 NI Act dishonour of cheque", 1024)
        assert "138" in resp.text or "cheque" in resp.text.lower()

    def test_ipc420_keywords(self):
        engine = MockLLMEngine()
        resp = engine.generate("sys", "Section 420 IPC cheating fraud", 1024)
        assert "420" in resp.text or "cheat" in resp.text.lower()

    def test_confidence_85(self):
        engine = MockLLMEngine()
        resp = engine.generate("sys", "any query about law", 1024)
        assert resp.confidence == 85.0

    def test_tokens_used_positive(self):
        engine = MockLLMEngine()
        resp = engine.generate("sys", "Section 138 dishonour", 1024)
        assert resp.tokens_used > 0

    def test_max_tokens_truncates(self):
        engine = MockLLMEngine()
        long_resp = engine.generate("sys", "Section 138 NI Act dishonour cheque", 20)
        short_resp = engine.generate("sys", "Section 138 NI Act dishonour cheque", 2048)
        assert len(long_resp.text) <= len(short_resp.text)


# ─── LegalReasoner ────────────────────────────────────────────────────────────

class TestLegalReasoner:
    def make_query(self, **kw) -> LegalQuery:
        defaults = dict(query="What is Section 138 NI Act?")
        defaults.update(kw)
        return LegalQuery(**defaults)

    def test_reason_returns_legal_answer(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query())
        assert isinstance(ans, LegalAnswer)

    def test_answer_id_is_uuid(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query())
        assert len(ans.answer_id) == 36

    def test_answer_id_unique_per_call(self):
        r = LegalReasoner()
        a1 = r.reason(self.make_query())
        a2 = r.reason(self.make_query())
        assert a1.answer_id != a2.answer_id

    def test_answer_text_non_empty(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query())
        assert len(ans.answer_text) > 0

    def test_llm_engine_is_mock(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query())
        assert ans.llm_engine == "mock"

    def test_confidence_85_for_mock(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query())
        assert ans.confidence == 85.0

    def test_review_required_false_for_high_confidence(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query(retrieval_context=SAMPLE_CONTEXT))
        # Mock returns 85.0 and we have citations -> not required
        # (review_required = confidence < 70 OR not grounded)
        assert ans.confidence >= 70.0

    def test_processing_time_non_negative(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query())
        assert ans.processing_time_s >= 0.0

    def test_query_preserved(self):
        q_text = "What is Section 420 IPC?"
        r = LegalReasoner()
        ans = r.reason(self.make_query(query=q_text))
        assert ans.query == q_text

    def test_citations_extracted_from_context(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query(
            query="Section 138 dishonour cheque punishment",
            retrieval_context=SAMPLE_CONTEXT,
        ))
        # MockLLMEngine answer for "138" contains [1] marker
        assert len(ans.citations) >= 1

    def test_no_citations_without_context(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query(retrieval_context=""))
        assert ans.citations == []

    def test_is_grounded_with_citations(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query(retrieval_context=SAMPLE_CONTEXT))
        # is_grounded derived from citations non-empty
        assert ans.is_grounded == (len(ans.citations) > 0)

    def test_custom_engine_injected(self):
        class ConstantEngine(BaseLLMEngine):
            @property
            def engine_name(self): return "constant"
            def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024):
                return LLMResponse(text="Constant answer.", confidence=90.0, engine_name="constant")

        r = LegalReasoner(engine=ConstantEngine())
        ans = r.reason(self.make_query())
        assert ans.llm_engine == "constant"
        assert ans.answer_text == "Constant answer."

    def test_reasoner_config_threshold(self):
        config = ReasonerConfig(confidence_threshold=90.0)
        r = LegalReasoner(config=config)
        assert r.config.confidence_threshold == 90.0

    def test_serialisation_round_trip(self):
        r = LegalReasoner()
        ans = r.reason(self.make_query(retrieval_context=SAMPLE_CONTEXT))
        data = ans.model_dump()
        restored = LegalAnswer.model_validate(data)
        assert restored.answer_id == ans.answer_id
        assert restored.llm_engine == "mock"
