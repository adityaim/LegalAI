"""
modules/llm/__init__.py
────────────────────────
Public API surface for Module 6: LLM Legal Reasoning.

Usage::

    from modules.llm import LegalReasoner, LegalQuery, LegalAnswer

    reasoner = LegalReasoner()   # MockLLMEngine by default (no API key needed)

    query = LegalQuery(
        query="What is the punishment under Section 138 NI Act?",
        fused_document_text=fused_doc.for_llm(),      # Module 4
        retrieval_context=retrieval_result.context_for_llm,  # Module 5
    )
    answer = reasoner.reason(query)

    print(answer.answer_text)
    print(answer.confidence)
    print(answer.citations)
"""

from .reasoner import LegalReasoner, ReasonerConfig
from .schema import LegalAnswer, LegalQuery, Citation

__all__ = [
    "LegalReasoner",
    "ReasonerConfig",
    "LegalAnswer",
    "LegalQuery",
    "Citation",
]
