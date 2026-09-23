"""
modules/rag/chunker.py
───────────────────────
Sentence-aware sliding-window chunker for Module 5 (RAG Ingestion).

Algorithm (mirrors LangChain's RecursiveCharacterTextSplitter approach,
but implemented cleanly without that dependency):

  1. Split text into sentences on strong sentence-ending punctuation.
  2. Greedily accumulate sentences into a window until token budget exceeded.
  3. Slide the window forward by (chunk_size - overlap) tokens.
  4. Assign source_modality by scanning which [TAG] block each sentence fell in.
  5. Mark has_review_flag if [REVIEW] appears anywhere in the chunk.

Parameters (from PipelineConfig):
  chunk_size   — target token count per chunk (default 512)
  overlap      — tokens shared with adjacent chunks (default 64)
  min_chunk    — minimum tokens; shorter chunks are merged with previous (default 32)

Reference: LangChain text_splitter.py (Apache-2.0)
https://github.com/langchain-ai/langchain/blob/master/libs/text_splitter.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import Chunk

# ─── Modality tag scanner ─────────────────────────────────────────────────────

# Maps source tag name → ModalitySource key
_TAG_TO_MODALITY: dict[str, str] = {
    "OCR-DOCUMENT":   "OCR",
    "HTR-HANDWRITING": "HTR",
    "ASR-VOICE":      "ASR",
    "TYPED-INPUT":    "TYPED",
}

_OPEN_TAG_RE  = re.compile(r"\[(OCR-DOCUMENT|HTR-HANDWRITING|ASR-VOICE|TYPED-INPUT)\]")
_CLOSE_TAG_RE = re.compile(r"\[/(OCR-DOCUMENT|HTR-HANDWRITING|ASR-VOICE|TYPED-INPUT)\]")
_REVIEW_RE    = re.compile(r"\[REVIEW\]")

# Sentence boundary: end on . ! ? followed by whitespace or end-of-string
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    """Approximate token count: split on whitespace (1 token ≈ 1 word)."""
    return len(text.split())


def _detect_modality_for_offset(text: str, char_offset: int) -> str:
    """
    Walk through *text* up to *char_offset* tracking [TAG] / [/TAG] pairs.
    Return the active modality tag at that position.
    """
    active = "UNKNOWN"
    for m in _OPEN_TAG_RE.finditer(text):
        if m.start() > char_offset:
            break
        active = _TAG_TO_MODALITY.get(m.group(1), "UNKNOWN")
    # If a close tag appears before offset, revert to UNKNOWN
    for m in _CLOSE_TAG_RE.finditer(text):
        if m.start() > char_offset:
            break
        active = "UNKNOWN"
    return active


# ─── Chunker ──────────────────────────────────────────────────────────────────

@dataclass
class ChunkerConfig:
    chunk_size: int = 512    # target tokens per chunk
    overlap: int = 64        # overlap tokens between adjacent chunks
    min_chunk: int = 32      # drop chunks shorter than this (merge into prev)


class SentenceAwareChunker:
    """
    Splits a FusedDocument's for_rag() text into overlapping Chunk objects.

    The chunker operates on the for_rag() output (source tags stripped) and
    uses character-offset tracking to determine which modality each sentence
    originated from (by consulting the original fused_text).
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(
        self,
        rag_text: str,
        fused_text: str,
        document_id: str,
    ) -> list[Chunk]:
        """
        Parameters
        ----------
        rag_text    : str
            Output of FusedDocument.for_rag() — source tags stripped.
        fused_text  : str
            Full FusedDocument.fused_text — used to resolve source modality.
        document_id : str
            UUID of the parent FusedDocument.

        Returns
        -------
        list[Chunk]
            Ordered list of Chunk objects ready for embedding and storage.
        """
        if not rag_text.strip():
            return []

        cfg = self.config
        sentences = _SENTENCE_RE.split(rag_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        start_char = 0
        char_cursor = 0   # track position in rag_text

        sentence_positions: list[tuple[str, int]] = []  # (sentence, start_char_in_rag)
        pos = 0
        for sent in sentences:
            idx = rag_text.find(sent, pos)
            if idx == -1:
                idx = pos
            sentence_positions.append((sent, idx))
            pos = idx + len(sent)

        sent_idx = 0
        chunk_index = 0

        while sent_idx < len(sentence_positions):
            sent, sent_start = sentence_positions[sent_idx]
            tok = _token_count(sent)

            # If adding this sentence exceeds budget, flush current window
            if current_tokens + tok > cfg.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                end_char = sent_start

                # Determine source modality from midpoint of chunk in fused_text
                mid_char = (start_char + end_char) // 2
                modality = self._resolve_modality(fused_text, rag_text, mid_char)

                if _token_count(chunk_text) >= cfg.min_chunk:
                    chunks.append(Chunk(
                        document_id=document_id,
                        source_modality=modality,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        has_review_flag=bool(_REVIEW_RE.search(chunk_text)),
                    ))
                    chunk_index += 1

                # Slide window: keep overlap tokens from the end
                overlap_sents: list[str] = []
                overlap_tokens = 0
                for prev_sent in reversed(current_sentences):
                    pt = _token_count(prev_sent)
                    if overlap_tokens + pt > cfg.overlap:
                        break
                    overlap_sents.insert(0, prev_sent)
                    overlap_tokens += pt

                current_sentences = overlap_sents
                current_tokens = overlap_tokens
                # Recalculate start_char for the overlap region
                if overlap_sents and overlap_sents[0] in rag_text:
                    start_char = rag_text.find(overlap_sents[0], max(0, start_char - 1))
                else:
                    start_char = sent_start

            current_sentences.append(sent)
            current_tokens += tok
            sent_idx += 1

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if _token_count(chunk_text) >= cfg.min_chunk:
                end_char = len(rag_text)
                mid_char = (start_char + end_char) // 2
                modality = self._resolve_modality(fused_text, rag_text, mid_char)
                chunks.append(Chunk(
                    document_id=document_id,
                    source_modality=modality,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    has_review_flag=bool(_REVIEW_RE.search(chunk_text)),
                ))
            elif chunks:
                # Merge tiny tail into last chunk
                chunks[-1] = chunks[-1].model_copy(update={
                    "text": chunks[-1].text + " " + chunk_text,
                    "end_char": len(rag_text),
                    "has_review_flag": chunks[-1].has_review_flag or bool(_REVIEW_RE.search(chunk_text)),
                })

        return chunks

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_modality(fused_text: str, rag_text: str, rag_char: int) -> str:
        """
        Map a character position in rag_text back to a modality in fused_text.

        Strategy: find the sentence at rag_char in rag_text, then search for
        a substring of it in fused_text, and identify the active [TAG] at that point.
        """
        # Extract 40-char snippet around the target position
        snippet_start = max(0, rag_char - 20)
        snippet = rag_text[snippet_start: snippet_start + 40].strip()
        if not snippet:
            return "UNKNOWN"
        fused_pos = fused_text.find(snippet[:20])
        if fused_pos == -1:
            return "UNKNOWN"
        return _detect_modality_for_offset(fused_text, fused_pos)
