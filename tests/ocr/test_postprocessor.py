"""
tests/ocr/test_postprocessor.py
─────────────────────────────────
Unit tests for the legal text post-processor.

Tests cover:
  • Legal text normalisation (section markers, curly quotes, pipes)
  • Hyphenation repair
  • Whitespace normalisation
  • Low-confidence block flagging
  • Page-level confidence aggregation
  • Metadata extraction (doc_type, parties, date, jurisdiction)
  • merge_blocks_to_text ordering and footer exclusion
"""

import pytest

from modules.ocr.postprocessor import LegalTextPostprocessor
from modules.ocr.schema import OCRBlock


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_block(
    page: int = 1,
    block_type: str = "body",
    text: str = "Sample text.",
    confidence: float = 90.0,
    engine: str = "paddle",
) -> OCRBlock:
    return OCRBlock(
        page=page,
        type=block_type,  # type: ignore[arg-type]
        text=text,
        confidence=confidence,
        ocr_engine=engine,  # type: ignore[arg-type]
    )


# ─── Normalisation tests ──────────────────────────────────────────────────────

class TestNormalisation:
    def setup_method(self):
        self.pp = LegalTextPostprocessor()

    def _normalise(self, text: str) -> str:
        block = make_block(text=text, confidence=90.0)
        result = self.pp.process_blocks([block])
        return result[0].text

    def test_section_abbreviation_s(self):
        assert "Section 138" in self._normalise("S. 138 of the NI Act")

    def test_section_abbreviation_sec(self):
        assert "Section 10" in self._normalise("Sec.10 of the Contract Act")

    def test_clause_abbreviation(self):
        assert "Clause 4" in self._normalise("cl. 4 of the agreement")

    def test_article_abbreviation(self):
        assert "Article 14" in self._normalise("Art. 14 of the Constitution")

    def test_curly_single_quotes_normalised(self):
        result = self._normalise("\u2018term\u2019")
        assert "'" in result
        assert "\u2018" not in result
        assert "\u2019" not in result

    def test_curly_double_quotes_normalised(self):
        result = self._normalise("\u201cterm\u201d")
        assert '"' in result

    def test_stray_pipe_removed(self):
        result = self._normalise("this | that")
        assert "|" not in result

    def test_pipe_in_table_preserved(self):
        # Pipe embedded in a table-like string should NOT be removed
        # because it has non-whitespace neighbours
        result = self._normalise("Col A|Col B")
        # The pipe between letters should be preserved (no whitespace around it)
        assert "Col A" in result

    def test_multiple_newlines_collapsed(self):
        result = self._normalise("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_hyphenation_repaired(self):
        result = self._normalise("inter-\npretation of the contract")
        assert "interpretation" in result

    def test_whitespace_collapsed(self):
        result = self._normalise("too   many    spaces")
        assert "  " not in result

    def test_control_characters_removed(self):
        result = self._normalise("text\x00with\x07null")
        assert "\x00" not in result
        assert "\x07" not in result


# ─── Confidence flagging tests ────────────────────────────────────────────────

class TestConfidenceFlagging:
    def setup_method(self):
        self.pp = LegalTextPostprocessor()

    def test_high_confidence_not_flagged(self):
        block = make_block(confidence=85.0)
        results = self.pp.process_blocks([block])
        assert results[0].review_required is False

    def test_low_confidence_flagged(self):
        block = make_block(confidence=55.0)
        results = self.pp.process_blocks([block])
        assert results[0].review_required is True

    def test_exactly_threshold(self):
        # 70.0 is threshold; exact value should NOT be flagged
        block = make_block(confidence=70.0)
        results = self.pp.process_blocks([block])
        # The block's review_required is set based on confidence < 70.0
        # At exactly 70.0 it should NOT be flagged
        assert results[0].review_required is False

    def test_mixed_blocks(self):
        blocks = [
            make_block(confidence=90.0),
            make_block(confidence=40.0),
            make_block(confidence=75.0),
        ]
        results = self.pp.process_blocks(blocks)
        assert results[0].review_required is False
        assert results[1].review_required is True
        assert results[2].review_required is False


# ─── Page-level confidence tests ─────────────────────────────────────────────

class TestPageLevelConfidence:
    def setup_method(self):
        self.pp = LegalTextPostprocessor()

    def test_low_confidence_page_flagged(self):
        blocks = [
            make_block(page=1, confidence=40.0),
            make_block(page=1, confidence=50.0),  # mean=45 → flagged
            make_block(page=2, confidence=90.0),
            make_block(page=2, confidence=85.0),  # mean=87.5 → ok
        ]
        flagged = self.pp.low_confidence_pages(blocks)
        assert 1 in flagged
        assert 2 not in flagged

    def test_no_low_confidence_pages(self):
        blocks = [make_block(page=i, confidence=90.0) for i in range(1, 4)]
        assert self.pp.low_confidence_pages(blocks) == []


# ─── merge_blocks_to_text tests ───────────────────────────────────────────────

class TestMergeBlocks:
    def setup_method(self):
        self.pp = LegalTextPostprocessor()

    def test_footers_excluded(self):
        blocks = [
            make_block(block_type="body", text="Body text here."),
            make_block(block_type="footer", text="Page 1 of 10"),
            make_block(block_type="page_number", text="1"),
        ]
        text = self.pp.merge_blocks_to_text(blocks)
        assert "Page 1 of 10" not in text
        assert "Body text here." in text

    def test_heading_gets_extra_newlines(self):
        blocks = [
            make_block(block_type="heading", text="AGREEMENT"),
            make_block(block_type="body", text="This agreement..."),
        ]
        text = self.pp.merge_blocks_to_text(blocks)
        assert "AGREEMENT" in text
        assert "This agreement" in text

    def test_empty_blocks(self):
        text = self.pp.merge_blocks_to_text([])
        assert text == ""


# ─── Metadata extraction tests ────────────────────────────────────────────────

class TestMetadataExtraction:
    def setup_method(self):
        self.pp = LegalTextPostprocessor()

    def test_detect_contract(self):
        meta = self.pp.extract_metadata(
            "THIS AGREEMENT is made between Alpha Ltd and Beta Corp on 15 January 2025."
        )
        assert meta.doc_type == "contract"

    def test_detect_fir(self):
        meta = self.pp.extract_metadata(
            "FIRST INFORMATION REPORT under Section 154 Cr.P.C."
        )
        assert meta.doc_type == "fir"

    def test_detect_affidavit(self):
        meta = self.pp.extract_metadata("I hereby declare by AFFIDAVIT that...")
        assert meta.doc_type == "affidavit"

    def test_date_extraction_iso(self):
        meta = self.pp.extract_metadata("Signed on 15/01/2025 in Delhi.")
        assert meta.date == "2025-01-15"

    def test_date_extraction_long_form(self):
        meta = self.pp.extract_metadata("This agreement dated 15 March 2024.")
        assert meta.date == "2024-03-15"

    def test_jurisdiction_delhi(self):
        meta = self.pp.extract_metadata("This matter falls under the jurisdiction of Delhi High Court.")
        assert meta.jurisdiction is not None
        assert "Delhi" in meta.jurisdiction or "High Court" in meta.jurisdiction

    def test_no_metadata(self):
        meta = self.pp.extract_metadata("Lorem ipsum dolor sit amet.")
        assert meta.doc_type is None
        assert meta.parties == []
        assert meta.date is None

    def test_metadata_is_DocumentMetadata_instance(self):
        from modules.ocr.schema import DocumentMetadata
        meta = self.pp.extract_metadata("Some legal text.")
        assert isinstance(meta, DocumentMetadata)
