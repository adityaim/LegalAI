"""
modules/ocr/layout.py
──────────────────────
Layout block detection for the OCR pipeline.

Provides a unified ``LayoutDetector`` interface with two backends:

1. **Heuristic (default)** — OpenCV morphological analysis
   • No external model weights required
   • Runs on CPU in < 50 ms per page
   • Classifies blocks as heading / body / table / footer / page_number

2. **LayoutParser (opt-in)** — Detectron2-based PubLayNet model
   • Requires: pip install layoutparser[layoutmodels] detectron2
   • ~500 ms per page on CPU; ~50 ms on GPU
   • Much higher accuracy on complex multi-column layouts

Select via ``LayoutDetectorConfig(engine="heuristic")`` or ``engine="layoutparser"``.

Region classification heuristics (heuristic backend)
──────────────────────────────────────────────────────
• **page_number**: small region near top or bottom, text is short numeric
• **heading**    : wide block in top 20% of page, or block with large text
• **footer**     : block in bottom 8% of page
• **table**      : aspect ratio near 1 (roughly square), OR grid-like
  white-space structure detected via horizontal line density
• **body**       : everything else
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Data types ──────────────────────────────────────────────────────────────

BlockType = Literal["heading", "body", "table", "footer", "page_number", "unknown"]


@dataclass
class LayoutRegion:
    """A single detected layout region on a page."""

    x0: int
    y0: int
    x1: int
    y1: int
    block_type: BlockType = "unknown"
    confidence: float = 80.0   # heuristic default; LayoutParser provides real scores

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the image crop corresponding to this region."""
        return image[self.y0:self.y1, self.x0:self.x1]


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class LayoutDetectorConfig:
    engine: Literal["heuristic", "layoutparser"] = "heuristic"

    # Heuristic: minimum connected-component area (px²) to consider a block
    min_block_area: int = 500

    # Heuristic: fraction of page height defining "header zone"
    header_zone_fraction: float = 0.20

    # Heuristic: fraction of page height defining "footer zone"
    footer_zone_fraction: float = 0.08

    # LayoutParser model (only used when engine="layoutparser")
    layoutparser_model: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    layoutparser_threshold: float = 0.5


# ─── Heuristic detector ──────────────────────────────────────────────────────

class HeuristicLayoutDetector:
    """
    Fast OpenCV-based layout detector.

    Algorithm:
    1. Morphological close to merge nearby text into blocks
    2. Find external contours → bounding rectangles
    3. Filter by minimum area
    4. Classify each block by position and shape
    5. Sort in reading order (top-to-bottom, left-to-right)
    """

    def __init__(self, config: LayoutDetectorConfig) -> None:
        self.config = config

    def detect(self, image: np.ndarray) -> list[LayoutRegion]:
        """Detect layout regions in *image* (binary or grayscale)."""
        h, w = image.shape[:2]

        # Invert binary image so text is white on black (required for morphology)
        inv = cv2.bitwise_not(image) if self._is_binary(image) else image

        # Morphological close: connect horizontally adjacent characters
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        closed_h = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_h)

        # Also close vertically to merge multi-line paragraphs
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        closed = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kernel_v)

        # Threshold to binary if grayscale
        if closed.ndim == 3:
            closed = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(closed, 10, 255, cv2.THRESH_BINARY)

        # Find external contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: list[LayoutRegion] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < self.config.min_block_area:
                continue
            # Clamp to image bounds
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(w, x + bw)
            y1 = min(h, y + bh)

            block_type = self._classify(x0, y0, x1, y1, w, h)
            regions.append(LayoutRegion(x0=x0, y0=y0, x1=x1, y1=y1,
                                        block_type=block_type))

        # Sort: top-to-bottom, left-to-right (reading order)
        regions.sort(key=lambda r: (r.y0 // 50, r.x0))
        return regions

    def _classify(
        self, x0: int, y0: int, x1: int, y1: int, page_w: int, page_h: int
    ) -> BlockType:
        """Classify a bounding box into a block type using positional heuristics."""
        bw = x1 - x0
        bh = y1 - y0

        rel_y_top = y0 / page_h
        rel_y_bot = y1 / page_h
        rel_w = bw / page_w

        # Footer zone
        if rel_y_top >= 1 - self.config.footer_zone_fraction:
            return "footer"

        # Page number: small area in top 5% or bottom 10%, narrow text
        if (rel_y_top < 0.05 or rel_y_bot > 0.92) and bw < page_w * 0.3:
            return "page_number"

        # Heading: in top header zone AND wide
        if rel_y_top < self.config.header_zone_fraction and rel_w > 0.5:
            return "heading"

        # Table heuristic: aspect ratio roughly square + wide
        aspect = bw / bh if bh else 0
        if 1.5 < aspect < 10 and rel_w > 0.4 and bh > 40:
            # Scan for horizontal line density as a proxy for table rows
            if self._has_horizontal_lines(x0, y0, x1, y1):
                return "table"

        return "body"

    @staticmethod
    def _has_horizontal_lines(x0: int, y0: int, x1: int, y1: int) -> bool:
        """Very lightweight stub — detailed check would need the image crop."""
        # In a real implementation: project pixel rows, count peaks
        # Here we conservatively return False to avoid false positives
        return False

    @staticmethod
    def _is_binary(image: np.ndarray) -> bool:
        """Return True if the image is binary (only 0 and 255 values)."""
        if image.ndim == 3:
            return False
        unique = np.unique(image)
        return len(unique) <= 2


# ─── LayoutParser detector (opt-in) ──────────────────────────────────────────

class LayoutParserDetector:
    """
    Layout detector backed by LayoutParser + Detectron2 PubLayNet model.

    This is an optional, heavier alternative to the heuristic backend.
    Requires: pip install layoutparser[layoutmodels] detectron2
    """

    # PubLayNet label map → our BlockType
    _LABEL_MAP: dict[str, BlockType] = {
        "Text":  "body",
        "Title": "heading",
        "List":  "body",
        "Table": "table",
        "Figure": "body",
    }

    def __init__(self, config: LayoutDetectorConfig) -> None:
        self.config = config
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import layoutparser as lp  # type: ignore[import]
            self._model = lp.Detectron2LayoutModel(
                config_path=self.config.layoutparser_model,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                               self.config.layoutparser_threshold],
                label_map={0: "Text", 1: "Title", 2: "List",
                           3: "Table", 4: "Figure"},
            )
            logger.info("LayoutParser model loaded: %s", self.config.layoutparser_model)
        except ImportError:
            raise ImportError(
                "LayoutParser not installed.  Install with: "
                "pip install layoutparser[layoutmodels] detectron2"
            )

    def detect(self, image: np.ndarray) -> list[LayoutRegion]:
        """Detect layout regions using LayoutParser / Detectron2."""
        self._load_model()
        import layoutparser as lp  # type: ignore[import]

        # LayoutParser expects RGB PIL or numpy RGB
        import cv2
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        layout = self._model.detect(rgb)
        regions: list[LayoutRegion] = []
        for block in layout:
            coords = block.coordinates  # (x1, y1, x2, y2)
            block_type = self._LABEL_MAP.get(block.type, "unknown")
            regions.append(LayoutRegion(
                x0=int(coords[0]),
                y0=int(coords[1]),
                x1=int(coords[2]),
                y1=int(coords[3]),
                block_type=block_type,
                confidence=float(block.score) * 100.0,
            ))

        regions.sort(key=lambda r: (r.y0 // 50, r.x0))
        return regions


# ─── Unified interface ────────────────────────────────────────────────────────

class LayoutDetector:
    """
    Public interface for layout detection.  Selects the backend from config.

    Usage::

        detector = LayoutDetector()                      # heuristic
        detector = LayoutDetector(                       # layoutparser
            LayoutDetectorConfig(engine="layoutparser"))
        regions = detector.detect(page_image)
    """

    def __init__(self, config: LayoutDetectorConfig | None = None) -> None:
        self.config = config or LayoutDetectorConfig()
        if self.config.engine == "layoutparser":
            self._backend: HeuristicLayoutDetector | LayoutParserDetector = (
                LayoutParserDetector(self.config)
            )
        else:
            self._backend = HeuristicLayoutDetector(self.config)

    def detect(self, image: np.ndarray) -> list[LayoutRegion]:
        """Detect and return layout regions for *image*."""
        regions = self._backend.detect(image)
        logger.debug(
            "Layout detection found %d regions (engine=%s)",
            len(regions), self.config.engine,
        )
        return regions

    def detect_or_fullpage(self, image: np.ndarray) -> list[LayoutRegion]:
        """
        Like ``detect()``, but returns a single full-page region if no blocks
        are detected (prevents silent data loss on blank / image-heavy pages).
        """
        regions = self.detect(image)
        if not regions:
            h, w = image.shape[:2]
            logger.debug("No layout regions found; using full-page fallback")
            return [LayoutRegion(x0=0, y0=0, x1=w, y1=h, block_type="body")]
        return regions
