"""
modules/ocr/preprocessor.py
────────────────────────────
Image pre-processing pipeline for the OCR module.

Implements the pipeline defined in Section 4.1 of the design document:
  1. Grayscale conversion
  2. Denoising (fast Non-Local Means)
  3. Binarisation (Otsu threshold)
  4. Skew detection and deskewing (Hough-based)
  5. Resolution normalisation (upscale if DPI < 150)

The output is always a binary (0/255) or grayscale NumPy uint8 array
suitable for passing directly to an OCR engine or layout detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .utils import (
    NDArray,
    compute_skew_angle,
    ensure_min_dpi,
    rotate_image,
)

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class PreprocessorConfig:
    """Knobs for the pre-processing pipeline. All flags default to *on*."""

    # NLM denoising strength.  Range 0–30; 0 disables denoising.
    denoise_h: int = 10

    # Otsu binarisation — set False to return grayscale instead of binary
    binarise: bool = True

    # Skew correction — set False to skip rotation
    deskew: bool = True

    # Maximum skew angle to correct (degrees).  Larger angles suggest
    # intentional landscape orientation, not skew.
    max_skew_angle: float = 15.0

    # DPI of the source image.  Used only for upscaling decisions.
    source_dpi: int = 300


# ─── Pre-processor ───────────────────────────────────────────────────────────

class ImagePreprocessor:
    """
    Stateless image pre-processing pipeline.

    Usage::

        pre = ImagePreprocessor()
        clean = pre.process(bgr_array, dpi=150)
    """

    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        self.config = config or PreprocessorConfig()

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, image: NDArray, dpi: int | None = None) -> NDArray:
        """
        Run the full pre-processing pipeline on *image*.

        Parameters
        ----------
        image : NDArray
            Input image — BGR (3-channel) or grayscale (1-channel) uint8.
        dpi : int | None
            Effective DPI of the source image.  If omitted, uses
            ``config.source_dpi``.

        Returns
        -------
        NDArray
            Pre-processed grayscale (or binary) uint8 array, same spatial
            dims as input after deskew (may differ slightly due to padding).
        """
        effective_dpi = dpi if dpi is not None else self.config.source_dpi

        # Step 1: ensure minimum resolution
        img = ensure_min_dpi(image, effective_dpi)

        # Step 2: convert to grayscale
        gray = self._to_gray(img)

        # Step 3: denoise
        gray = self._denoise(gray)

        # Step 4: binarise
        binary = self._binarise(gray)

        # Step 5: deskew
        result = self._deskew(binary)

        return result

    # ── Private steps ──────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(image: NDArray) -> NDArray:
        """Convert BGR or grayscale input to single-channel grayscale."""
        if image.ndim == 2:
            return image  # already grayscale
        if image.shape[2] == 4:
            # BGRA → BGR first
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _denoise(self, gray: NDArray) -> NDArray:
        """Apply fast Non-Local Means denoising."""
        h = self.config.denoise_h
        if h == 0:
            return gray
        denoised = cv2.fastNlMeansDenoising(gray, h=h)
        logger.debug("Denoising applied (h=%d)", h)
        return denoised

    def _binarise(self, gray: NDArray) -> NDArray:
        """Apply Otsu global threshold to produce a binary image."""
        if not self.config.binarise:
            return gray
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        logger.debug("Otsu binarisation applied")
        return binary

    def _deskew(self, binary: NDArray) -> NDArray:
        """Detect and correct skew using Hough-line angle estimation."""
        if not self.config.deskew:
            return binary
        angle = compute_skew_angle(binary, max_angle=self.config.max_skew_angle)
        if abs(angle) > 0.1:
            logger.info("Deskewing by %.2f°", angle)
            return rotate_image(binary, angle)
        return binary


# ─── Module-level convenience function ──────────────────────────────────────

_default_preprocessor = ImagePreprocessor()


def preprocess_page(image: NDArray, dpi: int = 300) -> NDArray:
    """
    Module-level convenience wrapper matching the design doc signature::

        clean_img = preprocess_page(image)

    Uses default ``PreprocessorConfig``.
    """
    return _default_preprocessor.process(image, dpi=dpi)
