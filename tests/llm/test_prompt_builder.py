"""
tests/llm/test_prompt_builder.py
──────────────────────────────────
Unit tests for Module 6 prompt builder.
"""

import pytest

from modules.llm.prompt_builder import build_system_prompt, build_user_prompt
from modules.llm.schema import LegalQuery


class TestBuildSystemPrompt:
    def test_returns_non_empty_string(self):
        p = build_system_prompt()
        assert isinstance(p, str)
        assert len(p) > 50

    def test_mentions_legal(self):
        p = build_system_prompt().lower()
        assert "legal" in p

    def test_mentions_context(self):
        p = build_system_prompt().lower()
        assert "context" in p

    def test_mentions_citation(self):
        p = build_system_prompt().lower()
        assert "cit" in p  # "cite" or "citation"

    def test_mentions_uncertain(self):
        assert "[UNCERTAIN]" in build_system_prompt()

    def test_consistent(self):
        # System prompt should be deterministic
        assert build_system_prompt() == build_system_prompt()


class TestBuildUserPrompt:
    def make_query(self, **kw) -> LegalQuery:
        defaults = dict(query="What is Section 138?")
        defaults.update(kw)
        return LegalQuery(**defaults)

    def test_query_included(self):
        q = self.make_query()
        p = build_user_prompt(q)
        assert "What is Section 138?" in p

    def test_legal_query_markers(self):
        q = self.make_query()
        p = build_user_prompt(q)
        assert "[LEGAL QUERY]" in p
        assert "[/LEGAL QUERY]" in p

    def test_fused_text_included_when_present(self):
        q = self.make_query(fused_document_text="[OCR-DOCUMENT]\nContract text\n[/OCR-DOCUMENT]")
        p = build_user_prompt(q)
        assert "Contract text" in p

    def test_retrieval_context_included_when_present(self):
        q = self.make_query(retrieval_context="[CONTEXT]\n[1] source=OCR score=0.9\nChunk text\n[/CONTEXT]")
        p = build_user_prompt(q)
        assert "[CONTEXT]" in p
        assert "Chunk text" in p

    def test_empty_fused_text_excluded(self):
        q = self.make_query(fused_document_text="")
        p = build_user_prompt(q)
        assert "[OCR-DOCUMENT]" not in p

    def test_empty_retrieval_context_excluded(self):
        q = self.make_query(retrieval_context="")
        p = build_user_prompt(q)
        # Should still have the query section
        assert "[LEGAL QUERY]" in p

    def test_all_context_combined(self):
        q = self.make_query(
            query="Section 138 NI Act?",
            fused_document_text="[OCR-DOCUMENT]\nDoc text\n[/OCR-DOCUMENT]",
            retrieval_context="[CONTEXT]\n[1] source=OCR score=0.9\nRetrieved chunk\n[/CONTEXT]",
        )
        p = build_user_prompt(q)
        assert "Doc text" in p
        assert "Retrieved chunk" in p
        assert "Section 138 NI Act?" in p

    def test_prompt_is_string(self):
        q = self.make_query()
        assert isinstance(build_user_prompt(q), str)

    def test_citation_instruction_present(self):
        q = self.make_query(retrieval_context="[CONTEXT]\n[1] source=OCR score=0.9\ntext\n[/CONTEXT]")
        p = build_user_prompt(q)
        assert "[1]" in p or "citation" in p.lower()
