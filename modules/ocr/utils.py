"""
modules/ocr/utils.py
─────────────────────
Shared image-manipulation helpers: skew detection, rotation,
DPI normalisation, and image-to-numpy conversions.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── DPI ─────────────────────────────────────────────────────────────────────

TARGET_DPI = 300          # design spec: rasterise at 300 DPI
MIN_DPI_FOR_OCR = 150     # below this we upscale


# ─── Type aliases ────────────────────────────────────────────────────────────

NDArray = np.ndarray  # BGR or grayscale uint8


# ─── Conversion helpers ──────────────────────────────────────────────────────

def pil_to_bgr(image: Image.Image) -> NDArray:
    """Convert a PIL Image (any mode) to a BGR NumPy array."""
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def bgr_to_pil(array: NDArray) -> Image.Image:
    """Convert a BGR NumPy array to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def gray_to_pil(array: NDArray) -> Image.Image:
    """Convert a grayscale NumPy array to a PIL Image (L mode)."""
    return Image.fromarray(array, mode="L")


# ─── Skew detection ──────────────────────────────────────────────────────────

def compute_skew_angle(binary: NDArray, max_angle: float = 15.0) -> float:
    """
    Estimate the skew angle of a binarised document image using a
    Hough-line-based approach.

    Parameters
    ----------
    binary : NDArray
        Binarised (0/255) grayscale image.
    max_angle : float
        Maximum skew angle to consider (degrees).  Beyond this the image
        is likely rotated intentionally (e.g. landscape page), not skewed.

    Returns
    -------
    float
        Estimated skew angle in degrees. Positive = counter-clockwise.
    """
    # Detect edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Probabilistic Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=100,
        minLineLength=binary.shape[1] // 4,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        logger.debug("compute_skew_angle: no lines detected, returning 0.0")
        return 0.0

    angles: list[float] = []
    for line in lines:
        # HoughLinesP returns shape (N, 1, 4) in older OpenCV
        # and (N, 4) in newer versions / NumPy combinations
        pts = line.flatten()
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
        if x2 == x1:
            continue  # vertical line — skip
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return 0.0

    # Median is more robust than mean against outlier lines
    median_angle = float(np.median(angles))
    logger.debug("Detected skew angle: %.2f°", median_angle)
    return median_angle


# ─── Image rotation ──────────────────────────────────────────────────────────

def rotate_image(image: NDArray, angle: float) -> NDArray:
    """
    Rotate *image* by *angle* degrees around its centre.

    Uses INTER_CUBIC for quality and pads borders with white (255) to avoid
    black margins after deskewing.
    """
    if abs(angle) < 0.1:
        return image  # no-op for negligible angles

    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)

    # Compute new bounding box after rotation
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - centre[0]
    M[1, 2] += (new_h / 2) - centre[1]

    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated


# ─── Resolution normalisation ────────────────────────────────────────────────

def ensure_min_dpi(image: NDArray, current_dpi: int) -> NDArray:
    """
    Upscale *image* if its effective DPI is below MIN_DPI_FOR_OCR.

    Uses Lanczos (INTER_LANCZOS4) for best quality preservation of thin
    strokes typical in legal document typefaces.
    """
    if current_dpi >= MIN_DPI_FOR_OCR:
        return image

    scale = MIN_DPI_FOR_OCR / current_dpi
    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)
    logger.info(
        "Upscaling image from %d DPI → %.0f DPI (factor %.2fx)",
        current_dpi,
        current_dpi * scale,
        scale,
    )
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


# ─── Misc ────────────────────────────────────────────────────────────────────

def mean_confidence(confidences: list[float]) -> float:
    """Return the mean of a list of confidence values, or 0.0 if empty."""
    valid = [c for c in confidences if c >= 0]
    return float(np.mean(valid)) if valid else 0.0
