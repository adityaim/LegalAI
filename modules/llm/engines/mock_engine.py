"""
modules/llm/engines/mock_engine.py
────────────────────────────────────
MockLLMEngine: deterministic, zero external dependencies.

Used in all tests and CI. Returns realistic legal answers based on keyword
matching without calling any real API.
"""

from __future__ import annotations

import re

from .base import BaseLLMEngine, LLMResponse

# ─── Canned answer templates ──────────────────────────────────────────────────

_NI_138_ANSWER = """\
Section 138 of the Negotiable Instruments Act, 1881 deals with the dishonour of cheques [1].

Where a cheque drawn by a person on an account maintained with a bank is returned unpaid, \
either because the amount of money standing to the credit of that account is insufficient to \
honour the cheque, or it exceeds the amount arranged to be paid, the drawer shall be deemed \
to have committed an offence [1].

Punishment: The offence is punishable with imprisonment up to two years, or with a fine up \
to twice the amount of the cheque, or with both [1].

Procedure: The payee must send a legal notice to the drawer within 30 days of the cheque \
being returned. The drawer then has 15 days to make payment. If payment is not made within \
that period, the payee may file a complaint within one month of the expiry of the 15-day \
notice period [1]."""

_IPC_420_ANSWER = """\
Section 420 of the Indian Penal Code, 1860 deals with cheating and dishonestly inducing \
delivery of property [1].

Whoever cheats and thereby dishonestly induces the person deceived to deliver any property \
to any person, or to make, alter or destroy the whole or any part of a valuable security or \
anything which is signed or sealed, shall be punished with imprisonment of either description \
for a term which may extend to seven years, and shall also be liable to fine [1].

Key element: Fraudulent intent must be established from the inception of the transaction. \
Mere breach of contract does not constitute cheating under IPC Section 420 [1]."""

_GENERIC_ANSWER = """\
Based on the provided context, the query relates to legal proceedings under Indian law [1].

[UNCERTAIN] The context does not contain sufficient specific information to provide a \
comprehensive answer to this query. Please ensure the relevant legal documents or \
statutory provisions are included in the context for a more precise response.

If you can provide additional context or specify the relevant statute and jurisdiction, \
a more precise answer can be given."""


def _score_keywords(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


class MockLLMEngine(BaseLLMEngine):
    """
    Deterministic mock engine for tests.

    Selects a canned legal answer based on keyword presence in the user prompt.
    Confidence is always 85.0 (above the 70.0 review threshold).
    """

    @property
    def engine_name(self) -> str:
        return "mock"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        text_lower = user_prompt.lower()

        ni138_score = _score_keywords(text_lower, ["138", "cheque", "dishonour", "dishonor", "negotiable"])
        ipc420_score = _score_keywords(text_lower, ["420", "cheat", "fraud", "delivery of property"])

        if ni138_score >= ipc420_score and ni138_score > 0:
            answer = _NI_138_ANSWER
        elif ipc420_score > 0:
            answer = _IPC_420_ANSWER
        else:
            answer = _GENERIC_ANSWER

        # Truncate to approximate max_tokens (1 token ~= 4 chars)
        max_chars = max_tokens * 4
        if len(answer) > max_chars:
            answer = answer[:max_chars].rsplit(".", 1)[0] + "."

        return LLMResponse(
            text=answer,
            confidence=85.0,
            engine_name=self.engine_name,
            tokens_used=len(answer.split()),
        )
