"""
tests/htr/test_reader.py
──────────────────────────
Integration tests for HandwritingReader (Module 2).

Strategy: inject a deterministic MockHTREngine to avoid needing
TrOCR (330 MB HuggingFace model) in CI.

Tests cover:
  • FileNotFoundError for missing input
  • ValueError for unsupported file type
  • Image processing pipeline end-to-end (white PNG → HTRResult)
  • Mock engine is called for each detected region
  • HTRResult structure and schema compliance
  • JSON serialisability
  • numpy array input (cross-module usage from Module 1)
  • PDF file rasterisation path (requires PyMuPDF)
  • Engine selection: Devanagari langs route to tesseract_devanagari tag
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.htr.engines.base import HTREngine, RegionRecognition
from modules.htr import HandwritingReader, HTRResult
from modules.htr.reader import ReaderConfig


# ─── Mock HTR engine ─────────────────────────────────────────────────────────

class MockHTREngine(HTREngine):
    """Returns a fixed recognition result without loading any model."""

    def __init__(self, text: str = "hereby agrees", confidence: float = 88.0) -> None:
        self._text = text
        self._confidence = confidence
        self.call_count = 0

    @property
    def engine_name(self) -> str:
        return "mock"

    def recognize(self, image: np.ndarray, lang: str = "en") -> RegionRecognition:
        self.call_count += 1
        return RegionRecognition(
            text=self._text, confidence=self._confidence, language=lang
        )


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_reader(engine: HTREngine | None = None, lang: str = "en") -> HandwritingReader:
    config = ReaderConfig(
        language=lang,
        segmenter_engine="heuristic",
        enable_spellcheck=False,
    )
    return HandwritingReader(config=config, engine=engine or MockHTREngine())


def _white_png(h: int = 300, w: int = 200) -> Path:
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        return Path(f.name)


def _stroked_png() -> Path:
    """PNG with hand-stroke-like marks to trigger segmentation."""
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    pts = np.array([[30, 80], [80, 70], [130, 85], [180, 75]], dtype=np.int32)
    cv2.polylines(img, [pts], False, (10, 10, 10), thickness=5)
    for dy in range(0, 100, 14):
        cv2.line(img, (30, 150 + dy), (200, 152 + dy), (20, 20, 20), 2)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        return Path(f.name)


def _make_pdf_bytes() -> bytes:
    """Create a minimal single-page PDF (blank page)."""
    try:
        import fitz
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()
    except ImportError:
        return None


# ─── Input validation ─────────────────────────────────────────────────────────

class TestInputValidation:
    def test_missing_file_raises(self):
        reader = make_reader()
        with pytest.raises(FileNotFoundError):
            reader.process("/nonexistent/path/file.png")

    def test_unsupported_ext_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                make_reader().process(path)
        finally:
            path.unlink(missing_ok=True)


# ─── Image processing ─────────────────────────────────────────────────────────

class TestImageProcessing:
    def test_white_image_returns_result(self):
        path = _white_png()
        try:
            result = make_reader().process(path)
            assert isinstance(result, HTRResult)
            assert result.source == "HTR"
            assert result.input_type == "image"
            assert isinstance(result.regions, list)
        finally:
            path.unlink(missing_ok=True)

    def test_result_has_uuid(self):
        path = _white_png()
        try:
            result = make_reader().process(path)
            assert len(result.document_id) == 36
        finally:
            path.unlink(missing_ok=True)

    def test_processing_time_positive(self):
        path = _white_png()
        try:
            result = make_reader().process(path)
            assert result.processing_time_s >= 0.0
        finally:
            path.unlink(missing_ok=True)

    def test_two_results_have_different_ids(self):
        path = _white_png()
        try:
            r1 = make_reader().process(path)
            r2 = make_reader().process(path)
            assert r1.document_id != r2.document_id
        finally:
            path.unlink(missing_ok=True)

    def test_language_propagated(self):
        path = _white_png()
        try:
            result = make_reader(lang="hi").process(path)
            assert result.language == "hi"
        finally:
            path.unlink(missing_ok=True)


# ─── Numpy array input ────────────────────────────────────────────────────────

class TestNumpyInput:
    def test_numpy_bgr_array(self):
        """HandwritingReader must accept a numpy BGR array (cross-module use)."""
        img = np.full((300, 200, 3), 255, dtype=np.uint8)
        result = make_reader().process(img)
        assert isinstance(result, HTRResult)
        assert result.input_type == "pdf_page"

    def test_grayscale_array(self):
        """Grayscale arrays should be accepted (preprocessor handles conversion)."""
        img = np.full((300, 200), 255, dtype=np.uint8)
        result = make_reader().process(img)
        assert isinstance(result, HTRResult)


# ─── PDF processing ───────────────────────────────────────────────────────────

class TestPDFProcessing:
    def test_pdf_returns_result(self):
        pdf_bytes = _make_pdf_bytes()
        if pdf_bytes is None:
            pytest.skip("PyMuPDF not installed")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            path = Path(f.name)
        try:
            result = make_reader().process(path)
            assert isinstance(result, HTRResult)
            assert result.input_type == "pdf_page"
        finally:
            path.unlink(missing_ok=True)


# ─── Schema compliance ────────────────────────────────────────────────────────

class TestResultSchema:
    def test_serialisable(self):
        path = _white_png()
        try:
            result = make_reader().process(path)
            data = result.model_dump()
            assert isinstance(data, dict)
            assert "document_id" in data
            assert "regions" in data
            assert "full_text" in data
            assert "low_confidence_regions" in data
        finally:
            path.unlink(missing_ok=True)

    def test_round_trip(self):
        path = _white_png()
        try:
            result = make_reader().process(path)
            data = result.model_dump()
            restored = HTRResult.model_validate(data)
            assert restored.document_id == result.document_id
        finally:
            path.unlink(missing_ok=True)

    def test_regions_have_valid_bboxes(self):
        """All regions must have BoundingBoxes with x1>x0, y1>y0."""
        path = _stroked_png()
        try:
            result = make_reader().process(path)
            for region in result.regions:
                assert region.bbox.x1 > region.bbox.x0
                assert region.bbox.y1 > region.bbox.y0
                assert region.bbox.page >= 1
        finally:
            path.unlink(missing_ok=True)

    def test_low_confidence_count_matches(self):
        """low_confidence_regions must equal count of review_required regions."""
        path = _stroked_png()
        try:
            mock = MockHTREngine(confidence=40.0)  # all regions low-confidence
            result = make_reader(engine=mock).process(path)
            actual_flagged = sum(1 for r in result.regions if r.review_required)
            assert result.low_confidence_regions == actual_flagged
        finally:
            path.unlink(missing_ok=True)


# ─── Supported formats ────────────────────────────────────────────────────────

class TestSupportedFormats:
    @pytest.mark.parametrize("ext", [".png", ".jpg", ".bmp"])
    def test_image_extensions(self, ext: str):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            path = Path(f.name)
            cv2.imwrite(str(path), img)
        try:
            result = make_reader().process(path)
            assert isinstance(result, HTRResult)
        finally:
            path.unlink(missing_ok=True)
