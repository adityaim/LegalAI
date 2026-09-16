"""
tests/ocr/test_schema.py
──────────────────────────
Unit tests for the OCR output schema (modules/ocr/schema.py).

Tests cover:
  • Model instantiation and field defaults
  • BoundingBox validation (x1 > x0, y1 > y0)
  • OCRBlock auto-derivation of review_required from confidence
  • OCRResult helper methods
  • JSON round-trip (model_dump → model_validate)
"""

import pytest
from pydantic import ValidationError

from modules.ocr.schema import BoundingBox, DocumentMetadata, OCRBlock, OCRResult


# ─── BoundingBox tests ────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_valid_bbox(self):
        bb = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0, page=1)
        assert bb.width == pytest.approx(90.0)
        assert bb.height == pytest.approx(180.0)
        assert bb.area == pytest.approx(90.0 * 180.0)

    def test_invalid_x1_le_x0(self):
        with pytest.raises(ValidationError):
            BoundingBox(x0=100.0, y0=0.0, x1=50.0, y1=100.0, page=1)

    def test_invalid_y1_le_y0(self):
        with pytest.raises(ValidationError):
            BoundingBox(x0=0.0, y0=100.0, x1=100.0, y1=50.0, page=1)

    def test_page_must_be_positive(self):
        with pytest.raises(ValidationError):
            BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0, page=0)

    def test_negative_coordinates_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(x0=-1.0, y0=0.0, x1=100.0, y1=100.0, page=1)


# ─── OCRBlock tests ───────────────────────────────────────────────────────────

class TestOCRBlock:
    def _block(self, **kwargs) -> OCRBlock:
        defaults = {
            "page": 1,
            "type": "body",
            "text": "This is a test paragraph.",
            "confidence": 95.0,
            "ocr_engine": "paddle",
        }
        defaults.update(kwargs)
        return OCRBlock(**defaults)

    def test_high_confidence_no_review(self):
        block = self._block(confidence=95.0)
        assert block.review_required is False

    def test_low_confidence_triggers_review(self):
        block = self._block(confidence=50.0)
        assert block.review_required is True

    def test_exactly_at_threshold(self):
        # 69.9 is below threshold (70.0) → should flag
        block_below = self._block(confidence=69.9)
        assert block_below.review_required is True
        # 70.0 exactly is NOT below threshold → no flag
        block_at = self._block(confidence=70.0)
        assert block_at.review_required is False

    def test_explicit_review_required_true_for_high_conf_block(self):
        # Model validator only ever sets True, never clears it.
        # Explicitly passing True for a high-conf block is fine.
        block = self._block(confidence=99.0, review_required=True)
        assert block.review_required is True

    def test_default_review_required_auto_derives_from_confidence(self):
        # When confidence < 70, review_required becomes True automatically
        block = self._block(confidence=40.0)
        assert block.review_required is True

    def test_block_types_are_valid(self):
        for block_type in ("heading", "body", "table", "footer", "page_number", "unknown"):
            b = self._block(type=block_type)
            assert b.type == block_type

    def test_invalid_block_type(self):
        with pytest.raises(ValidationError):
            self._block(type="sidebar")  # not in Literal

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            self._block(confidence=-1.0)
        with pytest.raises(ValidationError):
            self._block(confidence=101.0)

    def test_optional_bbox(self):
        block = self._block(bbox=None)
        assert block.bbox is None

        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50, page=1)
        block_with_bbox = self._block(bbox=bbox)
        assert block_with_bbox.bbox == bbox

    def test_serialisation_roundtrip(self):
        block = self._block(confidence=85.0)
        data = block.model_dump()
        restored = OCRBlock.model_validate(data)
        assert restored == block


# ─── DocumentMetadata tests ───────────────────────────────────────────────────

class TestDocumentMetadata:
    def test_defaults(self):
        m = DocumentMetadata()
        assert m.doc_type is None
        assert m.parties == []
        assert m.date is None
        assert m.jurisdiction is None

    def test_full_metadata(self):
        m = DocumentMetadata(
            doc_type="contract",
            parties=["Alpha Ltd", "Beta Corp"],
            date="2025-01-15",
            jurisdiction="Delhi",
        )
        assert m.doc_type == "contract"
        assert len(m.parties) == 2


# ─── OCRResult tests ──────────────────────────────────────────────────────────

def _make_result(**kwargs) -> OCRResult:
    defaults = {
        "pages": 2,
        "full_text": "THIS AGREEMENT is entered into by Alpha Ltd and Beta Corp.",
        "processing_time_s": 1.5,
    }
    defaults.update(kwargs)
    return OCRResult(**defaults)


class TestOCRResult:
    def test_auto_uuid(self):
        r = _make_result()
        assert len(r.document_id) == 36  # UUID4 format

    def test_source_is_ocr(self):
        r = _make_result()
        assert r.source == "OCR"

    def test_mean_confidence_empty(self):
        r = _make_result(blocks=[])
        assert r.mean_confidence() == 0.0

    def test_mean_confidence_with_blocks(self):
        blocks = [
            OCRBlock(page=1, type="body", text="abc", confidence=80.0, ocr_engine="paddle"),
            OCRBlock(page=1, type="body", text="def", confidence=60.0, ocr_engine="paddle"),
        ]
        r = _make_result(blocks=blocks)
        assert r.mean_confidence() == pytest.approx(70.0)

    def test_pages_needing_review(self):
        blocks = [
            OCRBlock(page=1, type="body", text="ok", confidence=90.0, ocr_engine="paddle"),
            OCRBlock(page=2, type="body", text="bad", confidence=40.0, ocr_engine="paddle",
                     review_required=True),
            OCRBlock(page=3, type="body", text="ok2", confidence=85.0, ocr_engine="paddle"),
        ]
        r = _make_result(blocks=blocks)
        assert r.pages_needing_review() == [2]

    def test_text_blocks_filter(self):
        blocks = [
            OCRBlock(page=1, type="heading", text="Title", confidence=95.0, ocr_engine="paddle"),
            OCRBlock(page=1, type="body", text="Body text", confidence=90.0, ocr_engine="paddle"),
            OCRBlock(page=1, type="footer", text="Page 1", confidence=99.0, ocr_engine="paddle"),
            OCRBlock(page=1, type="table", text="Col A | Col B", confidence=88.0, ocr_engine="paddle"),
        ]
        r = _make_result(blocks=blocks)
        text_only = r.text_blocks()
        types = {b.type for b in text_only}
        assert "footer" not in types
        assert "heading" in types
        assert "body" in types

    def test_json_roundtrip(self):
        blocks = [
            OCRBlock(page=1, type="body", text="test", confidence=75.0, ocr_engine="pymupdf_text"),
        ]
        r = _make_result(blocks=blocks, pages=1, low_confidence_pages=[])
        data = r.model_dump()
        restored = OCRResult.model_validate(data)
        assert restored.full_text == r.full_text
        assert restored.pages == r.pages
        assert restored.blocks[0].confidence == pytest.approx(75.0)

    def test_processing_time_non_negative(self):
        with pytest.raises(ValidationError):
            _make_result(processing_time_s=-0.1)
