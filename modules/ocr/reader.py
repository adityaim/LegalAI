"""
modules/ocr/reader.py
──────────────────────
OCRDocumentReader — the main pipeline orchestrator for Module 1.

Hybrid PDF handling strategy (Section 4.1):
────────────────────────────────────────────
  For each PDF page:
    1. Try PyMuPDF text extraction (fast, zero ML inference).
       If the page has a native text layer (≥ MIN_NATIVE_TEXT_CHARS chars),
       use it directly as ``pymupdf_text`` — no OCR needed.
    2. If the page is image-only (scanned) or has minimal text:
       a. Rasterise at TARGET_DPI (300) via PyMuPDF
       b. Pre-process (preprocess_page)
       c. Detect layout regions (LayoutDetector)
       d. For each region: run OCR engine (PaddleOCR by default)
       e. Collect WordResults → OCRBlock with bbox + confidence

  For image files (JPEG, PNG, TIFF, WebP):
    Direct path: load → pre-process → layout detect → OCR → assemble

Architecture diagram::

    process(path)
    ├── _is_pdf()
    │   ├── True  → _process_pdf()
    │   │            └── per page:
    │   │                ├── _extract_native_text()  ← PyMuPDF text layer
    │   │                └── _process_scanned_page() ← OCR path
    │   └── False → _process_image()
    │                └── _process_scanned_page()
    └── _assemble_result()

Thread safety: OCRDocumentReader instances are NOT thread-safe due to shared
PaddleOCR model state.  Use one instance per thread/worker in production.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .engines.base import OCREngine, WordResult
from .engines.paddle_engine import PaddleOCREngine
from .layout import LayoutDetector, LayoutDetectorConfig, LayoutRegion
from .postprocessor import LegalTextPostprocessor
from .preprocessor import ImagePreprocessor, PreprocessorConfig
from .schema import BoundingBox, DocumentMetadata, OCRBlock, OCRResult
from .utils import mean_confidence, pil_to_bgr

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────

TARGET_DPI = 300
MIN_NATIVE_TEXT_CHARS = 50   # pages with fewer chars are treated as scanned

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}
SUPPORTED_PDF_EXTS = {".pdf"}


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ReaderConfig:
    """Configuration for OCRDocumentReader."""

    # Primary OCR language (ISO 639-1); secondary is appended if needed
    language: str = "en"

    # OCR engine selection: "paddle" | "tesseract"
    ocr_engine: Literal["paddle", "tesseract"] = "paddle"

    # Layout detection engine: "heuristic" | "layoutparser"
    layout_engine: Literal["heuristic", "layoutparser"] = "heuristic"

    # DPI for PDF rasterisation
    rasterise_dpi: int = TARGET_DPI

    # Minimum native text chars to skip OCR on a PDF page
    min_native_chars: int = MIN_NATIVE_TEXT_CHARS

    # Pre-processing config
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)

    # Layout detector config
    layout: LayoutDetectorConfig = field(default_factory=LayoutDetectorConfig)


# ─── Main reader ─────────────────────────────────────────────────────────────

class OCRDocumentReader:
    """
    Main OCR pipeline class.  Accepts PDF or image file paths and returns
    a fully-populated ``OCRResult``.

    Usage::

        reader = OCRDocumentReader()
        result = reader.process("contract.pdf")
        print(result.full_text[:500])
        print(result.mean_confidence())
    """

    def __init__(
        self,
        config: ReaderConfig | None = None,
        engine: OCREngine | None = None,
    ) -> None:
        self.config = config or ReaderConfig()

        # OCR engine (injectable for testing / swapping)
        if engine is not None:
            self._engine = engine
        elif self.config.ocr_engine == "tesseract":
            from .engines.tesseract_engine import TesseractEngine
            self._engine: OCREngine = TesseractEngine(
                default_lang=self.config.language
            )
        else:
            self._engine = PaddleOCREngine(
                default_lang=self.config.language
            )

        self._preprocessor = ImagePreprocessor(self.config.preprocessor)

        layout_cfg = self.config.layout
        layout_cfg.engine = self.config.layout_engine
        self._layout = LayoutDetector(layout_cfg)

        self._postprocessor = LegalTextPostprocessor()

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, path: str | Path) -> OCRResult:
        """
        Process a PDF or image file and return a structured ``OCRResult``.

        Parameters
        ----------
        path : str | Path
            Absolute or relative path to the input file.

        Returns
        -------
        OCRResult
            Fully populated result including text, blocks, metadata, and
            processing statistics.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file extension is not supported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_PDF_EXTS | SUPPORTED_IMAGE_EXTS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {sorted(SUPPORTED_PDF_EXTS | SUPPORTED_IMAGE_EXTS)}"
            )

        start = time.perf_counter()
        doc_id = str(uuid.uuid4())
        logger.info("Processing document: %s (id=%s)", path.name, doc_id)

        if ext in SUPPORTED_PDF_EXTS:
            blocks, page_count = self._process_pdf(path)
        else:
            blocks, page_count = self._process_image(path)

        # Post-processing
        blocks = self._postprocessor.process_blocks(blocks)
        full_text = self._postprocessor.merge_blocks_to_text(blocks)
        metadata = self._postprocessor.extract_metadata(full_text)
        low_conf_pages = self._postprocessor.low_confidence_pages(blocks)

        # Language detection
        language = self._detect_language(full_text)

        elapsed = time.perf_counter() - start
        logger.info(
            "Processed %d pages, %d blocks in %.2fs (mean conf=%.1f%%)",
            page_count, len(blocks), elapsed,
            sum(b.confidence for b in blocks) / len(blocks) if blocks else 0,
        )

        return OCRResult(
            document_id=doc_id,
            source="OCR",
            language=language,
            pages=page_count,
            full_text=full_text,
            blocks=blocks,
            metadata=metadata,
            processing_time_s=round(elapsed, 3),
            low_confidence_pages=low_conf_pages,
        )

    # ── PDF processing ────────────────────────────────────────────────────

    def _process_pdf(self, path: Path) -> tuple[list[OCRBlock], int]:
        """Process a PDF file; returns (blocks, page_count)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF not installed.  Install with: pip install pymupdf"
            )

        doc = fitz.open(str(path))
        all_blocks: list[OCRBlock] = []

        for page_num, page in enumerate(doc, start=1):
            logger.debug("Processing PDF page %d/%d", page_num, len(doc))

            # Try native text extraction first (hybrid strategy)
            native_text = page.get_text("text").strip()

            if len(native_text) >= self.config.min_native_chars:
                # Page has usable text layer — skip OCR
                logger.debug("Page %d: using native text layer (%d chars)",
                             page_num, len(native_text))
                page_blocks = self._native_text_to_blocks(
                    native_text, page, page_num
                )
            else:
                # Page is scanned — rasterise and OCR
                logger.debug("Page %d: native text too short (%d chars), running OCR",
                             page_num, len(native_text))
                matrix = fitz.Matrix(self.config.rasterise_dpi / 72, self.config.rasterise_dpi / 72)
                pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, 3
                )
                # PyMuPDF returns RGB; convert to BGR for OpenCV
                import cv2
                bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                page_blocks = self._process_scanned_page(bgr, page_num)

            all_blocks.extend(page_blocks)

        page_count = len(doc)
        doc.close()
        return all_blocks, page_count

    def _native_text_to_blocks(
        self, text: str, page, page_num: int  # noqa: ANN001
    ) -> list[OCRBlock]:
        """
        Convert PyMuPDF native text extraction to OCRBlock list.

        PyMuPDF ``get_text("blocks")`` returns structured block data with
        bounding boxes.  We use this for richer output instead of raw text.
        """
        raw_blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,block_no,block_type)
        result_blocks: list[OCRBlock] = []

        for raw in raw_blocks:
            if len(raw) < 5:
                continue
            x0, y0, x1, y1, block_text = raw[0], raw[1], raw[2], raw[3], raw[4]
            block_text = block_text.strip()
            if not block_text:
                continue

            block_type = self._classify_native_block(block_text, y0, page.rect.height)

            result_blocks.append(OCRBlock(
                page=page_num,
                type=block_type,
                text=block_text,
                confidence=99.0,  # native text is essentially perfect
                bbox=BoundingBox(
                    x0=float(x0), y0=float(y0),
                    x1=float(x1), y1=float(y1),
                    page=page_num,
                ),
                review_required=False,
                ocr_engine="pymupdf_text",
            ))

        return result_blocks

    @staticmethod
    def _classify_native_block(text: str, y0: float, page_height: float) -> str:
        """Classify a native PyMuPDF block by position and content."""
        rel_y = y0 / page_height if page_height else 0.5

        if rel_y > 0.92:
            return "footer"
        if rel_y < 0.05 and len(text) < 50:
            return "page_number"

        # Heuristic: if text is short and in the top 20%, likely heading
        if rel_y < 0.20 and len(text) < 200 and text.isupper():
            return "heading"
        if rel_y < 0.20 and len(text) < 100:
            return "heading"

        return "body"

    # ── Image processing ──────────────────────────────────────────────────

    def _process_image(self, path: Path) -> tuple[list[OCRBlock], int]:
        """Load a single image file and run OCR on it."""
        from PIL import Image
        pil_img = Image.open(str(path))
        bgr = pil_to_bgr(pil_img)
        blocks = self._process_scanned_page(bgr, page_num=1)
        return blocks, 1

    # ── Core OCR path ─────────────────────────────────────────────────────

    def _process_scanned_page(
        self, image: np.ndarray, page_num: int
    ) -> list[OCRBlock]:
        """
        Run the full OCR pipeline on a single scanned page image.

        Pipeline:
          preprocess → layout detect → per-region OCR → assemble blocks
        """
        # Step 1: pre-process
        clean = self._preprocessor.process(image, dpi=self.config.rasterise_dpi)

        # Step 2: detect layout regions
        regions: list[LayoutRegion] = self._layout.detect_or_fullpage(clean)

        # Step 3: OCR each region
        blocks: list[OCRBlock] = []
        for region in regions:
            crop = region.crop(clean)
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue

            words: list[WordResult] = self._engine.recognize(
                crop, lang=self.config.language
            )

            if not words:
                continue

            text = self._engine.words_to_text(words)
            conf = self._engine.mean_confidence(words)

            # Translate region-local bbox to page-level bbox
            page_bbox = BoundingBox(
                x0=float(region.x0),
                y0=float(region.y0),
                x1=float(region.x1),
                y1=float(region.y1),
                page=page_num,
            )

            # Map engine name to a valid OCREngineTag; custom engines map to "unknown"
            _VALID_ENGINE_TAGS = {"pymupdf_text", "paddle", "tesseract", "unknown"}
            engine_tag = self._engine.name if self._engine.name in _VALID_ENGINE_TAGS else "unknown"

            blocks.append(OCRBlock(
                page=page_num,
                type=region.block_type,
                text=text,
                confidence=conf,
                bbox=page_bbox,
                review_required=False,      # model_validator will set True if conf < 70
                ocr_engine=engine_tag,      # type: ignore[arg-type]
            ))

        return blocks

    # ── Language detection ────────────────────────────────────────────────

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect the dominant language of *text* using langdetect."""
        if not text or len(text) < 20:
            return "en"
        try:
            from langdetect import detect  # type: ignore[import]
            return detect(text[:2000])  # use first 2000 chars for speed
        except Exception:
            return "en"
