"""
modules/llm/reasoner.py
────────────────────────
LegalReasoner — the main Module 6 pipeline.

Pipeline:
    LegalQuery
        -> PromptBuilder (system + user prompts)
        -> BaseLLMEngine.generate()
        -> Citation extraction from retrieval context
        -> LegalAnswer

Citation extraction:
    Parses the [CONTEXT] block in retrieval_context for numbered entries:
        [N] source=OCR score=0.923
        <chunk text...>
    Then scans answer_text for [N] references and creates Citation objects
    for each matched chunk.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from .engines.base import BaseLLMEngine
from .engines.mock_engine import MockLLMEngine
from .prompt_builder import build_system_prompt, build_user_prompt
from .schema import Citation, LegalAnswer, LegalQuery

logger = logging.getLogger(__name__)

# Matches lines like:  [3] source=ASR score=0.712
_CONTEXT_ENTRY_RE = re.compile(
    r"\[(\d+)\]\s+source=(\S+)\s+score=([\d.]+)",
    re.IGNORECASE,
)

# Matches [N] citation references in answer text (greedy: [1], [2], [1][3] etc.)
_CITATION_REF_RE = re.compile(r"\[(\d+)\]")


@dataclass
class ReasonerConfig:
    """Configuration for LegalReasoner."""
    confidence_threshold: float = 70.0   # below this → review_required
    max_tokens: int = 1024
    extract_citations: bool = True        # parse [CONTEXT] block for citations


# ─── Citation extraction ──────────────────────────────────────────────────────

def _parse_context_chunks(retrieval_context: str) -> dict[int, dict]:
    """
    Parse the [CONTEXT] block produced by RetrievalResult.context_for_llm.

    Returns a dict mapping citation number (1-based int) to a dict with keys:
      source_modality, score, text_excerpt
    """
    chunks: dict[int, dict] = {}
    if not retrieval_context.strip():
        return chunks

    # Extract lines between [CONTEXT] and [/CONTEXT]
    ctx_match = re.search(r"\[CONTEXT\](.*?)\[/CONTEXT\]", retrieval_context, re.DOTALL)
    if not ctx_match:
        # Try to parse without the wrapper
        ctx_body = retrieval_context
    else:
        ctx_body = ctx_match.group(1)

    lines = ctx_body.splitlines()
    current_num: int | None = None
    current_meta: dict = {}
    current_text_lines: list[str] = []

    def _flush():
        if current_num is not None and current_meta:
            text = " ".join(current_text_lines).strip()
            chunks[current_num] = {
                "source_modality": current_meta.get("source", "UNKNOWN"),
                "score": float(current_meta.get("score", 0.0)),
                "text_excerpt": text[:200],
            }

    for line in lines:
        m = _CONTEXT_ENTRY_RE.match(line.strip())
        if m:
            _flush()
            current_num = int(m.group(1))
            current_meta = {"source": m.group(2), "score": m.group(3)}
            current_text_lines = []
        elif current_num is not None and line.strip() not in ("[CONTEXT]", "[/CONTEXT]"):
            current_text_lines.append(line)

    _flush()
    return chunks


def _extract_citations(
    answer_text: str,
    retrieval_context: str,
    document_id: str = "unknown",
) -> list[Citation]:
    """
    Build a Citation list from [N] references found in answer_text.

    Each [N] in the answer is matched to the corresponding chunk in
    the retrieval context. Duplicates are deduplicated (by citation number).
    """
    context_chunks = _parse_context_chunks(retrieval_context)
    if not context_chunks:
        return []

    # Find all citation numbers referenced in answer
    cited_nums: set[int] = set()
    for m in _CITATION_REF_RE.finditer(answer_text):
        cited_nums.add(int(m.group(1)))

    citations: list[Citation] = []
    for num in sorted(cited_nums):
        chunk_meta = context_chunks.get(num)
        if chunk_meta is None:
            continue
        citations.append(Citation(
            chunk_id=f"ctx-chunk-{num}",
            document_id=document_id,
            source_modality=chunk_meta["source_modality"],
            text_excerpt=chunk_meta["text_excerpt"],
            relevance_score=min(1.0, max(0.0, chunk_meta["score"])),
        ))

    return citations


# ─── LegalReasoner ────────────────────────────────────────────────────────────

class LegalReasoner:
    """
    Main pipeline for Module 6: LLM Legal Reasoning.

    Usage::

        reasoner = LegalReasoner()   # MockLLMEngine by default
        answer = reasoner.reason(legal_query)
    """

    def __init__(
        self,
        engine: BaseLLMEngine | None = None,
        config: ReasonerConfig | None = None,
    ) -> None:
        self.engine = engine or MockLLMEngine()
        self.config = config or ReasonerConfig()

    def reason(self, query: LegalQuery) -> LegalAnswer:
        """
        Run the full reasoning pipeline.

        1. Build prompts.
        2. Call LLM engine.
        3. Extract citations from retrieval context.
        4. Build and return LegalAnswer.
        """
        t0 = time.perf_counter()

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(query)

        llm_response = self.engine.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=query.max_tokens,
        )

        citations: list[Citation] = []
        if self.config.extract_citations and query.retrieval_context:
            citations = _extract_citations(
                answer_text=llm_response.text,
                retrieval_context=query.retrieval_context,
            )

        elapsed = time.perf_counter() - t0

        answer = LegalAnswer(
            query=query.query,
            answer_text=llm_response.text,
            citations=citations,
            confidence=llm_response.confidence,
            language=query.language,
            detected_intent=query.detected_intent,
            processing_time_s=round(elapsed, 4),
            llm_engine=self.engine.engine_name,
        )

        logger.info(
            "LegalReasoner: engine=%s, confidence=%.1f, citations=%d, grounded=%s, time=%.3fs",
            answer.llm_engine, answer.confidence, len(answer.citations),
            answer.is_grounded, elapsed,
        )

        return answer
