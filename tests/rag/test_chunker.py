"""
tests/rag/test_chunker.py
──────────────────────────
Unit tests for the SentenceAwareChunker (Module 5).
"""

import pytest

from modules.rag.chunker import SentenceAwareChunker, ChunkerConfig
from modules.rag.schema import Chunk


LEGAL_TEXT = (
    "Section 138 of the Negotiable Instruments Act 1881 deals with dishonour of cheque. "
    "Where any cheque drawn by a person is returned by the bank unpaid, the drawer is deemed to have committed an offence. "
    "The offence is punishable with imprisonment up to two years or fine up to twice the amount of the cheque. "
    "A legal notice must be sent within 30 days of cheque dishonour. "
    "The accused must make payment within 15 days of receiving the legal notice. "
    "The Supreme Court has interpreted this provision strictly in several landmark judgments. "
    "The High Court of Delhi has held that the notice period is mandatory and cannot be waived. "
    "Failure to comply with the notice requirement renders the complaint maintainable. "
    "The drawer of the cheque bears the burden of proving sufficient funds were available."
)

FUSED_TEXT = f"[OCR-DOCUMENT]\n{LEGAL_TEXT}\n[/OCR-DOCUMENT]"


class TestSentenceAwareChunker:
    def make_chunker(self, chunk_size=100, overlap=20, min_chunk=5) -> SentenceAwareChunker:
        return SentenceAwareChunker(ChunkerConfig(
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk=min_chunk,
        ))

    def test_produces_chunks(self):
        chunker = self.make_chunker()
        chunks = chunker.chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        assert len(chunks) >= 1

    def test_chunk_type(self):
        chunks = self.make_chunker().chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_document_id_propagated(self):
        chunks = self.make_chunker().chunk(LEGAL_TEXT, FUSED_TEXT, "test-doc")
        for c in chunks:
            assert c.document_id == "test-doc"

    def test_chunk_ids_unique(self):
        chunks = self.make_chunker().chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_indices_sequential(self):
        chunks = self.make_chunker(chunk_size=30).chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_empty_text_returns_empty(self):
        chunker = self.make_chunker()
        chunks = chunker.chunk("", FUSED_TEXT, "doc-001")
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunker = self.make_chunker()
        chunks = chunker.chunk("   \n   ", FUSED_TEXT, "doc-001")
        assert chunks == []

    def test_no_review_flag_by_default(self):
        chunks = self.make_chunker().chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        assert all(not c.has_review_flag for c in chunks)

    def test_review_flag_detected(self):
        text_with_review = "Normal text. [REVIEW]unclear part[/REVIEW] More text here."
        fused = f"[OCR-DOCUMENT]\n{text_with_review}\n[/OCR-DOCUMENT]"
        chunks = self.make_chunker(chunk_size=20, min_chunk=2).chunk(text_with_review, fused, "doc")
        flagged = [c for c in chunks if c.has_review_flag]
        assert len(flagged) >= 1

    def test_small_chunk_size_creates_multiple_chunks(self):
        chunks = self.make_chunker(chunk_size=20, overlap=5).chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        assert len(chunks) >= 3

    def test_large_chunk_size_creates_fewer_chunks(self):
        chunks_small = self.make_chunker(chunk_size=30).chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        chunks_large = self.make_chunker(chunk_size=500).chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        assert len(chunks_large) <= len(chunks_small)

    def test_start_end_chars_non_negative(self):
        chunks = self.make_chunker().chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        for c in chunks:
            assert c.start_char >= 0
            assert c.end_char >= 0

    def test_text_not_empty(self):
        chunks = self.make_chunker().chunk(LEGAL_TEXT, FUSED_TEXT, "doc-001")
        for c in chunks:
            assert len(c.text.strip()) > 0

    def test_multimodality_tags(self):
        multi_fused = (
            "[OCR-DOCUMENT]\nOCR text about Section 138.\n[/OCR-DOCUMENT]\n\n"
            "[ASR-VOICE]\nASR transcript about payment terms.\n[/ASR-VOICE]"
        )
        multi_rag = "OCR text about Section 138. ASR transcript about payment terms."
        chunks = self.make_chunker(chunk_size=10, min_chunk=2).chunk(
            multi_rag, multi_fused, "doc-multi"
        )
        assert len(chunks) >= 1
        # Modalities should be strings
        for c in chunks:
            assert isinstance(c.source_modality, str)
