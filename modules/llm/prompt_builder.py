"""
modules/llm/prompt_builder.py
──────────────────────────────
Builds the system + user prompts sent to the LLM engine.

The prompt design follows the "grounded legal assistant" pattern:
  • System: role + strict instructions (ground answers, cite, flag uncertainty)
  • User:   fused document context + RAG retrieval context + query

References
──────────
LangChain legal QA prompt (MIT):
https://github.com/langchain-ai/langchain/tree/master/cookbook/legal_qa
"""

from __future__ import annotations

from .schema import LegalQuery

# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise, expert Indian legal assistant specialising in:
- Negotiable Instruments Act (NI Act)
- Indian Penal Code (IPC)
- Code of Criminal Procedure (CrPC)
- Civil Procedure Code (CPC)
- Indian Contract Act
- Specific Relief Act
- Supreme Court and High Court precedents

STRICT INSTRUCTIONS:
1. GROUND every statement in the provided context. Do NOT invent case law, sections, or legal provisions.
2. CITE sources using [1], [2], ... markers that correspond to the numbered chunks in [CONTEXT].
3. Flag any statement you are uncertain about with [UNCERTAIN] immediately before it.
4. If the context is insufficient to answer, say so explicitly: "The provided context does not contain enough information to answer this question definitively."
5. Use precise, formal legal language.
6. Structure longer answers with clear paragraphs.
7. When quoting statute text, reproduce it verbatim from the context.
"""


def build_system_prompt() -> str:
    """Return the system / role prompt for the legal LLM."""
    return _SYSTEM_PROMPT.strip()


# ─── User prompt ──────────────────────────────────────────────────────────────

def build_user_prompt(query: LegalQuery) -> str:
    """
    Assemble the user-facing prompt from all context sources.

    Structure:
        [optional fused document section]
        [optional RAG retrieval context]
        [LEGAL QUERY] ... [/LEGAL QUERY]
        Instruction line
    """
    parts: list[str] = []

    if query.fused_document_text.strip():
        parts.append(query.fused_document_text.strip())

    if query.retrieval_context.strip():
        parts.append(query.retrieval_context.strip())

    parts.append(
        f"[LEGAL QUERY]\n{query.query.strip()}\n[/LEGAL QUERY]\n\n"
        "Provide a precise, grounded legal answer with numbered citations [1], [2], ... "
        "where each number corresponds to the matching chunk in [CONTEXT] above."
    )

    return "\n\n".join(parts)
