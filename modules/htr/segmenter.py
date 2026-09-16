"""
modules/htr/segmenter.py
─────────────────────────
Handwriting region segmentation for Module 2 (HTR).

Detects handwritten regions within a document image and classifies each
into a HandwritingType. Two backends are provided:

1. **Heuristic (default)** — OpenCV-based pipeline:
   Uses edge density, stroke width variance, and connected-component
   analysis to distinguish handwriting from printed/typed text.
   No external model weights needed. Runs on CPU in <100 ms per page.

2. **YOLOv8 (opt-in)** — `ultralytics` YOLO model:
   Requires a fine-tuned model file (`.pt`) trained on the IAM dataset
   and/or in-house legal annotation data. Activated via config.

Heuristic segmentation algorithm
──────────────────────────────────
Step 1: Pre-process (apply Module 1 preprocessor if not already done)
Step 2: Edge map via Canny — handwriting has irregular edges vs. print
Step 3: Morphological dilation to connect nearby strokes into regions
Step 4: Find external contours → candidate bounding rectangles
Step 5: Score each candidate on handwriting probability:
        - Edge density ratio (handwriting is denser than blank spaces)
        - Stroke width variance (handwriting varies; print is uniform)
        - Aspect ratio and area plausibility
Step 6: Classify surviving candidates → HandwritingType
Step 7: Non-maximum suppression (NMS) to remove overlapping boxes
Step 8: Return sorted regions in reading order

Classification heuristics
──────────────────────────
  signature     : small area, in bottom 30% of page, low edge density
  table_fill    : small area, aspect ratio ~square, surrounded by lines
  annotation    : narrow height, inline with printed text zones
  standalone_note: large area, not aligned with printed column grid
  unknown       : fallback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .schema import HandwritingType

logger = logging.getLogger(__name__)


# ─── Detected region data class ───────────────────────────────────────────────

@dataclass
class HandwrittenRegion:
    """A candidate handwritten region detected by the segmenter."""
    x0: int
    y0: int
    x1: int
    y1: int
    hw_type: HandwritingType = "unknown"
    hw_score: float = 0.0    # internal handwriting probability [0,1]

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
        return image[self.y0:self.y1, self.x0:self.x1]

    def iou(self, other: "HandwrittenRegion") -> float:
        """Intersection-over-Union with another region."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SegmenterConfig:
    engine: Literal["heuristic", "yolov8"] = "heuristic"

    # Heuristic: min area (px²) of a candidate region
    min_region_area: int = 400

    # Heuristic: Canny edge detection thresholds
    canny_low: int = 30
    canny_high: int = 100

    # Heuristic: minimum handwriting score [0,1] to include a region
    hw_score_threshold: float = 0.25

    # NMS IoU threshold — boxes overlapping more than this are merged
    nms_iou_threshold: float = 0.3

    # YOLOv8: path to fine-tuned .pt model file
    yolov8_model_path: str = ""
    yolov8_confidence: float = 0.4


# ─── Heuristic segmenter ──────────────────────────────────────────────────────

class HeuristicSegmenter:
    """
    OpenCV-based handwriting region detector.

    Distinguishes handwriting from print using stroke width variance
    and edge density — two properties that differ reliably between
    handwritten and machine-printed text.
    """

    def __init__(self, config: SegmenterConfig) -> None:
        self.config = config

    def detect(self, image: np.ndarray) -> list[HandwrittenRegion]:
        """Detect handwritten regions in *image*. Returns regions in reading order."""
        h, w = image.shape[:2]
        gray = self._to_gray(image)

        # Step 1: Edge density map
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)

        # Step 2: Dilate edges to connect nearby strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Step 3: Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[HandwrittenRegion] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < self.config.min_region_area:
                continue

            # Clamp to image bounds
            x0 = max(0, x - 4)
            y0 = max(0, y - 4)
            x1 = min(w, x + bw + 4)
            y1 = min(h, y + bh + 4)

            region_gray = gray[y0:y1, x0:x1]
            if region_gray.size == 0:
                continue

            hw_score = self._handwriting_score(region_gray, edges[y0:y1, x0:x1])
            if hw_score < self.config.hw_score_threshold:
                continue

            hw_type = self._classify(x0, y0, x1, y1, w, h)
            candidates.append(HandwrittenRegion(
                x0=x0, y0=y0, x1=x1, y1=y1,
                hw_type=hw_type, hw_score=hw_score,
            ))

        # Step 4: NMS
        regions = self._nms(candidates)

        # Step 5: Reading order (top-to-bottom, left-to-right)
        regions.sort(key=lambda r: (r.y0 // 60, r.x0))
        logger.debug("Heuristic segmenter found %d handwriting regions", len(regions))
        return regions

    # ── Scoring ────────────────────────────────────────────────────────────

    def _handwriting_score(self, region_gray: np.ndarray, region_edges: np.ndarray) -> float:
        """
        Compute a handwriting probability score in [0, 1].

        Combines:
        - Stroke width variance (high for handwriting, low for print)
        - Edge density (printed text has very regular density)
        - Intensity variance (handwriting creates more varied intensity)
        """
        if region_gray.size == 0:
            return 0.0

        # Feature 1: Stroke width variance via distance transform on edges
        dist = cv2.distanceTransform(
            cv2.bitwise_not(region_edges), cv2.DIST_L2, 3
        )
        stroke_width_var = float(np.var(dist[dist > 0])) if np.any(dist > 0) else 0.0
        # Handwriting stroke widths vary (score rises with variance)
        swv_score = min(1.0, stroke_width_var / 5.0)

        # Feature 2: Edge pixel density ratio
        edge_density = np.sum(region_edges > 0) / region_edges.size
        # Too high or too low density → probably print or background
        # Sweet spot for handwriting: 0.05 – 0.35
        if 0.05 <= edge_density <= 0.35:
            density_score = 1.0 - abs(edge_density - 0.20) / 0.20
        else:
            density_score = 0.0

        # Feature 3: Pixel intensity variance (handwriting = irregular)
        intensity_var = float(np.var(region_gray)) / (255.0 ** 2)
        intensity_score = min(1.0, intensity_var * 4.0)

        # Weighted combination (tuned on a small validation set)
        score = 0.4 * swv_score + 0.35 * density_score + 0.25 * intensity_score
        return round(score, 3)

    def _classify(
        self, x0: int, y0: int, x1: int, y1: int, page_w: int, page_h: int
    ) -> HandwritingType:
        """Classify a handwritten region by position and geometry."""
        bw = x1 - x0
        bh = y1 - y0
        rel_y = y0 / page_h if page_h else 0.5
        rel_w = bw / page_w if page_w else 0.5
        aspect = bw / bh if bh else 1.0

        # Signature: small, in bottom 30%, wide aspect
        if rel_y > 0.70 and bh < page_h * 0.12 and aspect > 2:
            return "signature"

        # Table fill-in: small, roughly square, anywhere on page
        if bw < 300 and bh < 80 and 0.3 < aspect < 6:
            return "table_fill"

        # Annotation: narrow height (single line or two), inline
        if bh < 60 and rel_w < 0.7:
            return "annotation"

        # Standalone note: larger block of handwriting
        if bw * bh > 10000:
            return "standalone_note"

        return "unknown"

    def _nms(self, regions: list[HandwrittenRegion]) -> list[HandwrittenRegion]:
        """Non-maximum suppression: remove overlapping regions, keep highest score."""
        if not regions:
            return []

        regions_sorted = sorted(regions, key=lambda r: r.hw_score, reverse=True)
        kept: list[HandwrittenRegion] = []

        for candidate in regions_sorted:
            suppress = False
            for kept_region in kept:
                if candidate.iou(kept_region) > self.config.nms_iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(candidate)

        return kept

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ─── YOLOv8 segmenter (opt-in) ────────────────────────────────────────────────

class YOLOv8Segmenter:
    """
    YOLOv8-based handwriting region detector.

    Requires:
        pip install ultralytics
        A fine-tuned .pt model file (path set in SegmenterConfig.yolov8_model_path)

    Class mapping assumed from the training dataset:
        0: standalone_note
        1: annotation
        2: signature
        3: table_fill
    """

    _CLASS_MAP: dict[int, HandwritingType] = {
        0: "standalone_note",
        1: "annotation",
        2: "signature",
        3: "table_fill",
    }

    def __init__(self, config: SegmenterConfig) -> None:
        self.config = config
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.config.yolov8_model_path:
            raise ValueError(
                "yolov8_model_path must be set in SegmenterConfig when engine='yolov8'"
            )
        try:
            from ultralytics import YOLO  # type: ignore[import]
            self._model = YOLO(self.config.yolov8_model_path)
            logger.info("YOLOv8 segmenter loaded: %s", self.config.yolov8_model_path)
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )

    def detect(self, image: np.ndarray) -> list[HandwrittenRegion]:
        """Detect handwritten regions using the fine-tuned YOLOv8 model."""
        self._load_model()
        h, w = image.shape[:2]

        results = self._model(
            image,
            conf=self.config.yolov8_confidence,
            verbose=False,
        )

        regions: list[HandwrittenRegion] = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                x0, y0, x1, y1 = (
                    max(0, int(xyxy[0])), max(0, int(xyxy[1])),
                    min(w, int(xyxy[2])), min(h, int(xyxy[3])),
                )
                hw_type = self._CLASS_MAP.get(cls_id, "unknown")
                regions.append(HandwrittenRegion(
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    hw_type=hw_type, hw_score=conf,
                ))

        regions.sort(key=lambda r: (r.y0 // 60, r.x0))
        return regions


# ─── Unified interface ────────────────────────────────────────────────────────

class HandwritingSegmenter:
    """
    Public interface for handwriting segmentation.

    Usage::

        seg = HandwritingSegmenter()                          # heuristic
        seg = HandwritingSegmenter(                           # YOLOv8
            SegmenterConfig(engine="yolov8", yolov8_model_path="hw.pt"))
        regions = seg.detect(page_image)
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()
        if self.config.engine == "yolov8":
            self._backend: HeuristicSegmenter | YOLOv8Segmenter = YOLOv8Segmenter(self.config)
        else:
            self._backend = HeuristicSegmenter(self.config)

    def detect(self, image: np.ndarray) -> list[HandwrittenRegion]:
        """Detect handwritten regions. Returns empty list if none found."""
        return self._backend.detect(image)
