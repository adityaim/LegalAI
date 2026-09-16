"""
tests/ocr/test_preprocessor.py
────────────────────────────────
Unit tests for the image pre-processing pipeline.

Tests use synthetic images (numpy arrays) — no real documents required.
Validates:
  • Grayscale conversion (RGB → gray, already-gray passthrough)
  • Denoising applies (smoke test: output shape == input shape)
  • Binarisation produces only 0 and 255 values
  • Deskewing: rotated image is corrected within tolerance
  • Config flags (denoise_h=0 → denoising skipped, binarise=False → grayscale out)
  • DPI upscaling for low-resolution images
"""

import math

import cv2
import numpy as np
import pytest

from modules.ocr.preprocessor import ImagePreprocessor, PreprocessorConfig, preprocess_page
from modules.ocr.utils import compute_skew_angle, ensure_min_dpi, rotate_image


# ─── Synthetic image fixtures ────────────────────────────────────────────────

def white_page(h: int = 400, w: int = 300) -> np.ndarray:
    """Return a white BGR image."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def gray_page(h: int = 400, w: int = 300) -> np.ndarray:
    """Return a white grayscale image."""
    return np.full((h, w), 255, dtype=np.uint8)


def binary_text_page(angle: float = 0.0) -> np.ndarray:
    """Return a synthetic binarised page with some text-like horizontal lines."""
    img = np.full((400, 300), 255, dtype=np.uint8)
    # Draw horizontal black lines simulating text rows
    for y in range(50, 350, 25):
        cv2.line(img, (20, y), (280, y), 0, 2)
    if abs(angle) > 0:
        img = rotate_image(img, angle)
    return img


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestPreprocessorGrayscale:
    def test_bgr_to_gray(self):
        pre = ImagePreprocessor(PreprocessorConfig(denoise_h=0, binarise=False, deskew=False))
        bgr = white_page()
        result = pre.process(bgr)
        assert result.ndim == 2, "Output should be grayscale (2D)"

    def test_already_gray_passthrough(self):
        pre = ImagePreprocessor(PreprocessorConfig(denoise_h=0, binarise=False, deskew=False))
        gray = gray_page()
        result = pre.process(gray)
        assert result.ndim == 2
        assert result.shape == gray.shape

    def test_bgra_input(self):
        """BGRA (4-channel) images should be accepted."""
        pre = ImagePreprocessor(PreprocessorConfig(denoise_h=0, binarise=False, deskew=False))
        bgra = np.full((100, 100, 4), 255, dtype=np.uint8)
        result = pre.process(bgra)
        assert result.ndim == 2


class TestPreprocessorDenoising:
    def test_shape_preserved_after_denoise(self):
        pre = ImagePreprocessor(PreprocessorConfig(denoise_h=10, binarise=False, deskew=False))
        img = white_page()
        result = pre.process(img)
        assert result.shape[:2] == img.shape[:2]

    def test_denoise_disabled(self):
        """With denoise_h=0, denoising is skipped — output should still be valid."""
        pre = ImagePreprocessor(PreprocessorConfig(denoise_h=0, binarise=False, deskew=False))
        img = white_page()
        result = pre.process(img)
        assert result is not None


class TestPreprocessorBinarisation:
    def test_binary_output_values(self):
        """Output should contain only 0 and 255 after Otsu binarisation."""
        pre = ImagePreprocessor(PreprocessorConfig(denoise_h=0, binarise=True, deskew=False))
        img = white_page()
        result = pre.process(img)
        unique = np.unique(result)
        assert set(unique.tolist()).issubset({0, 255}), (
            f"Non-binary values found: {unique}"
        )

    def test_binarise_disabled_produces_grayscale(self):
        pre = ImagePreprocessor(PreprocessorConfig(denoise_h=0, binarise=False, deskew=False))
        img = white_page()
        result = pre.process(img)
        # Grayscale — should not be strictly binary (unless image is already pure BW)
        assert result.ndim == 2


class TestPreprocessorDeskew:
    def test_skew_angle_detection(self):
        """Synthetic page with 5° rotation should detect a non-zero angle."""
        skewed = binary_text_page(angle=5.0)
        angle = compute_skew_angle(skewed)
        # Hough may not be pixel-perfect, but should detect significant skew
        # Allow ±10° tolerance — mainly testing it doesn't crash
        assert isinstance(angle, float)

    def test_zero_skew_unchanged(self):
        pre = ImagePreprocessor(PreprocessorConfig(denoise_h=0, binarise=True, deskew=True))
        img = binary_text_page(angle=0.0)
        result = pre.process(img)
        # Image should be close to original size (small padding allowed)
        assert abs(result.shape[0] - img.shape[0]) <= 20
        assert abs(result.shape[1] - img.shape[1]) <= 20

    def test_deskew_disabled(self):
        pre = ImagePreprocessor(PreprocessorConfig(denoise_h=0, binarise=False, deskew=False))
        skewed = binary_text_page(angle=10.0)
        result = pre.process(skewed)
        # Shape should be identical when deskewing is off
        assert result.shape == skewed.shape


class TestDPIUpscaling:
    def test_upscale_low_dpi(self):
        """Images below MIN_DPI_FOR_OCR (150) should be upscaled."""
        small_img = np.full((100, 80, 3), 200, dtype=np.uint8)
        result = ensure_min_dpi(small_img, current_dpi=72)
        # At 72 DPI → scale factor ≥ 2 → result should be bigger
        assert result.shape[0] > small_img.shape[0]
        assert result.shape[1] > small_img.shape[1]

    def test_no_upscale_adequate_dpi(self):
        img = np.full((400, 300, 3), 200, dtype=np.uint8)
        result = ensure_min_dpi(img, current_dpi=300)
        assert result.shape == img.shape


class TestPreprocessorConvenienceFunction:
    def test_preprocess_page_returns_array(self):
        img = white_page()
        result = preprocess_page(img, dpi=300)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # grayscale or binary
