"""
modules/htr/reader.py
──────────────────────
HandwritingReader — the main pipeline orchestrator for Module 2 (HTR).

Pipeline overview::

    process(path | image_array)
    ├── load_image()                  # PIL / numpy from file or array
    ├── preprocess_page()             # reuse Module 1 pre-processor
    ├── segmenter.detect()            # find handwriting regions (heuristic / YOLOv8)
    ├── For each region:
    │   ├── engine_for_lang()         # pick TrOCR or Tesseract-Devanagari
    │   ├── engine.recognize(crop)    # run HTR
    │   └── build HTRRegion
    ├── postprocessor.process_regions()   # SymSpell + confidence flagging
    ├── postprocessor.assemble_text()     # join with [REVIEW] markers
    └── assemble HTRResult()

Engine selection strategy
──────────────────────────
  lang == "hi" / "mr" / "ne" / any Devanagari → TesseractDevanagariEngine
  lang == "en" / other Latin script           → TrOCREngine (primary)

Fallback: if the primary engine returns empty text (model not loaded,
low-res crop, etc.), the region is still included in the result with
confidence = 0 and review_required = True.

Reuse from Module 1
────────────────────
  • modules.ocr.preprocessor.preprocess_page   — image cleaning
  • modules.ocr.schema.BoundingBox             — used inside HTRRegion
  • modules.ocr.utils.pil_to_bgr               — PIL → numpy conversion
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from modules.ocr.preprocessor import preprocess_page
from modules.ocr.schema import BoundingBox
from modules.ocr.utils import pil_to_bgr

from .engines.base import HTREngine, RegionRecognition
from .engines.trocr_engine import TrOCREngine
from .engines.tesseract_devanagari import TesseractDevanagariEngine
from .postprocessor import HTRPostprocessor
from .schema import HTRRegion, HTRResult
from .segmenter import HandwritingSegmenter, SegmenterConfig

logger = logging.getLogger(__name__)

# ─── Supported inputs ─────────────────────────────────────────────────────────

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}
SUPPORTED_PDF_EXTS   = {".pdf"}

# Devanagari / Hindi language codes → Tesseract Devanagari engine
_DEVANAGARI_LANGS = {"hi", "mr", "ne", "sa"}


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ReaderConfig:
    """Configuration for HandwritingReader."""

    # Primary language (ISO 639-1)
    language: str = "en"

    # Segmentation backend: "heuristic" (default) | "yolov8"
    segmenter_engine: Literal["heuristic", "yolov8"] = "heuristic"

    # Path to YOLOv8 model (only needed when segmenter_engine="yolov8")
    yolov8_model_path: str = ""

    # TrOCR model variant: "base" | "large" | "small"
    trocr_variant: Literal["base", "large", "small"] = "base"

    # Compute device for TrOCR: "cpu" | "cuda"
    device: str = "cpu"

    # Enable SymSpell spell correction
    enable_spellcheck: bool = True

    # Minimum region area to process (px²)
    min_region_area: int = 400


# ─── TrOCR model name map ─────────────────────────────────────────────────────

_TROCR_MODEL_MAP = {
    "base":  "microsoft/trocr-base-handwritten",
    "large": "microsoft/trocr-large-handwritten",
    "small": "microsoft/trocr-small-handwritten",
}


# ─── Main reader ─────────────────────────────────────────────────────────────

class HandwritingReader:
    """
    Main HTR pipeline. Accepts image files, numpy arrays, or PDF page images.

    Usage::

        reader = HandwritingReader()
        result = reader.process("annotated_contract.png")
        print(result.full_text)
        print(result.low_confidence_regions, "regions need review")
    """

    def __init__(
        self,
        config: ReaderConfig | None = None,
        engine: HTREngine | None = None,
    ) -> None:
        self.config = config or ReaderConfig()

        # Injected engine (used in tests) or build from config
        self._engine_override = engine

        # Segmenter
        seg_cfg = SegmenterConfig(
            engine=self.config.segmenter_engine,
            yolov8_model_path=self.config.yolov8_model_path,
            min_region_area=self.config.min_region_area,
        )
        self._segmenter = HandwritingSegmenter(seg_cfg)

        # Post-processor
        self._postprocessor = HTRPostprocessor(
            enable_spellcheck=self.config.enable_spellcheck
        )

        # Cached engines (lazy-loaded on first use)
        self._trocr: TrOCREngine | None = None
        self._tesseract_dev: TesseractDevanagariEngine | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, source: "str | Path | np.ndarray") -> HTRResult:
        """
        Detect and recognise handwritten regions in *source*.

        Parameters
        ----------
        source : str | Path | np.ndarray
            File path (image or single-page PDF raster) OR a BGR numpy array
            (e.g. a page already rasterised by Module 1).

        Returns
        -------
        HTRResult
            All detected handwritten regions with text, confidence, bbox,
            and region type.
        """
        start = time.perf_counter()
        doc_id = str(uuid.uuid4())
        input_type = "image"

        # ── Load image ─────────────────────────────────────────────────────
        if isinstance(source, np.ndarray):
            bgr = source
            input_type = "pdf_page"
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            ext = path.suffix.lower()
            if ext not in SUPPORTED_IMAGE_EXTS | SUPPORTED_PDF_EXTS:
                raise ValueError(
                    f"Unsupported file type '{ext}'. "
                    f"Supported: {sorted(SUPPORTED_IMAGE_EXTS | SUPPORTED_PDF_EXTS)}"
                )
            if ext in SUPPORTED_PDF_EXTS:
                bgr = self._rasterise_pdf_first_page(path)
                input_type = "pdf_page"
            else:
                bgr = self._load_image(path)

        # ── Pre-process ────────────────────────────────────────────────────
        clean = preprocess_page(bgr, dpi=300)
        h, w = clean.shape[:2]

        # ── Segment ────────────────────────────────────────────────────────
        raw_regions = self._segmenter.detect(clean)
        logger.info("Segmenter found %d handwriting candidates", len(raw_regions))

        # ── Recognise each region ──────────────────────────────────────────
        htr_regions: list[HTRRegion] = []
        for seg_region in raw_regions:
            crop = seg_region.crop(clean)
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            engine = self._get_engine(self.config.language)
            recognition: RegionRecognition = engine.recognize(
                crop, lang=self.config.language
            )

            if recognition.is_empty():
                # Include as a flagged empty region rather than silently dropping
                recognition = RegionRecognition(
                    text="", confidence=0.0, language=self.config.language
                )

            bbox = BoundingBox(
                x0=float(seg_region.x0),
                y0=float(seg_region.y0),
                x1=float(seg_region.x1),
                y1=float(seg_region.y1),
                page=1,
            )

            engine_tag = self._engine_tag(engine)
            htr_regions.append(HTRRegion(
                page=1,
                type=seg_region.hw_type,
                text=recognition.text,
                confidence=recognition.confidence,
                bbox=bbox,
                review_required=False,  # model_validator will auto-set
                htr_engine=engine_tag,
                language=recognition.language,
            ))

        # ── Post-process ───────────────────────────────────────────────────
        htr_regions = self._postprocessor.process_regions(htr_regions)
        full_text = self._postprocessor.assemble_text(htr_regions)
        low_conf_count = sum(1 for r in htr_regions if r.review_required)

        elapsed = time.perf_counter() - start
        logger.info(
            "HTR complete: %d regions, %d flagged, %.2fs",
            len(htr_regions), low_conf_count, elapsed,
        )

        return HTRResult(
            document_id=doc_id,
            source="HTR",
            input_type=input_type,
            language=self.config.language,
            regions=htr_regions,
            full_text=full_text,
            processing_time_s=round(elapsed, 3),
            low_confidence_regions=low_conf_count,
        )

    # ── Engine selection ──────────────────────────────────────────────────

    def _get_engine(self, lang: str) -> HTREngine:
        """Select the appropriate HTR engine for the requested language."""
        if self._engine_override is not None:
            return self._engine_override

        if lang in _DEVANAGARI_LANGS:
            if self._tesseract_dev is None:
                self._tesseract_dev = TesseractDevanagariEngine(default_lang=lang)
            return self._tesseract_dev

        # Default: TrOCR
        if self._trocr is None:
            model_name = _TROCR_MODEL_MAP.get(self.config.trocr_variant,
                                               _TROCR_MODEL_MAP["base"])
            self._trocr = TrOCREngine(
                model_name=model_name, device=self.config.device
            )
        return self._trocr

    @staticmethod
    def _engine_tag(engine: HTREngine) -> str:
        """Map engine to a valid HTREngineTag; unknown engines → 'unknown'."""
        valid = {"trocr", "tesseract_devanagari", "mock", "unknown"}
        name = engine.engine_name
        return name if name in valid else "unknown"

    # ── Image loading helpers ─────────────────────────────────────────────

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        from PIL import Image
        pil = Image.open(str(path))
        return pil_to_bgr(pil)

    @staticmethod
    def _rasterise_pdf_first_page(path: Path) -> np.ndarray:
        """Rasterise the first page of a PDF at 300 DPI and return BGR array."""
        try:
            import fitz
            import cv2
            doc = fitz.open(str(path))
            page = doc[0]
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            page_count = len(doc)
            doc.close()
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except ImportError:
            raise ImportError(
                "PyMuPDF not installed. Install with: pip install pymupdf"
            )
