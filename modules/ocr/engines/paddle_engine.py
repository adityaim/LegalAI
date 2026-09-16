"""
modules/ocr/engines/paddle_engine.py
──────────────────────────────────────
PaddleOCR engine adapter — primary OCR engine for the Legal AI system.

PaddleOCR is chosen over Tesseract as the primary engine because:
  • Superior accuracy on Hindi / Devanagari text
  • Built-in text detection + recognition pipeline (end-to-end)
  • Better handling of low-quality / skewed scans
  • Actively maintained with strong multilingual support

Language code mapping
─────────────────────
PaddleOCR uses its own language codes.  We map ISO 639-1 codes to Paddle
equivalents in ``LANG_MAP``.  Unsupported codes fall back to English.

Lazy loading
─────────────
The PaddleOCR model is large (~200 MB) and slow to initialise.  We load
it lazily on the first ``recognize()`` call, or eagerly via ``warmup()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import OCREngine, WordResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── Language mapping ────────────────────────────────────────────────────────

# ISO 639-1 → PaddleOCR language code
LANG_MAP: dict[str, str] = {
    "en": "en",
    "hi": "hi",          # Hindi / Devanagari
    "mr": "hi",          # Marathi (uses Devanagari script — same model)
    "bn": "ta",          # Bengali — closest available; upgrade when Paddle adds bn
    "te": "te",          # Telugu
    "ta": "ta",          # Tamil
    "gu": "gu",          # Gujarati
    "pa": "en",          # Punjabi — fallback to English (not yet in Paddle)
    "zh": "ch",          # Chinese Simplified (unlikely but mapped for completeness)
}

DEFAULT_PADDLE_LANG = "en"


# ─── Engine ──────────────────────────────────────────────────────────────────

class PaddleOCREngine(OCREngine):
    """
    OCR engine backed by PaddleOCR.

    Parameters
    ----------
    default_lang : str
        ISO 639-1 code for the default language (``"en"`` or ``"hi"``).
    use_gpu : bool
        Whether to run inference on GPU.  Defaults to False (CPU) to allow
        development without a GPU; set True in production Docker image.
    use_angle_cls : bool
        Enable text-direction classification (handles 180° rotated text).
        Adds ~20 ms per page; recommended for scanned legal documents.
    """

    def __init__(
        self,
        default_lang: str = "en",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
    ) -> None:
        self._default_lang = default_lang
        self._use_gpu = use_gpu
        self._use_angle_cls = use_angle_cls
        self._ocr_instances: dict[str, object] = {}  # lang → PaddleOCR instance
        self._paddle_available: bool | None = None

    # ── Public interface ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "paddle"

    def warmup(self) -> None:
        """Pre-load the model for the default language."""
        self._get_ocr(self._default_lang)
        logger.info("PaddleOCR warmed up for lang='%s'", self._default_lang)

    def recognize(self, image: np.ndarray, lang: str = "en") -> list[WordResult]:
        """
        Run PaddleOCR on *image*.

        Confidence values returned by PaddleOCR are already in 0–1 range;
        we multiply by 100 to match the 0–100 schema convention.

        If PaddleOCR is not installed, returns an empty list and logs a
        warning rather than raising — the reader falls back gracefully.
        """
        ocr = self._get_ocr(lang)
        if ocr is None:
            return []

        # PaddleOCR expects a BGR or RGB uint8 array.
        # Grayscale (2D) must be converted to 3-channel.
        img = self._ensure_3ch(image)

        try:
            results = ocr.ocr(img, cls=self._use_angle_cls)
        except Exception as exc:
            logger.error("PaddleOCR.ocr() raised: %s", exc)
            return []

        return self._parse_results(results)

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_ocr(self, iso_lang: str) -> object | None:
        """Return a cached PaddleOCR instance for *iso_lang*, creating if needed."""
        paddle_lang = LANG_MAP.get(iso_lang, DEFAULT_PADDLE_LANG)

        if paddle_lang not in self._ocr_instances:
            try:
                from paddleocr import PaddleOCR  # type: ignore[import]
                instance = PaddleOCR(
                    use_angle_cls=self._use_angle_cls,
                    lang=paddle_lang,
                    use_gpu=self._use_gpu,
                    show_log=False,
                )
                self._ocr_instances[paddle_lang] = instance
                self._paddle_available = True
            except ImportError:
                if self._paddle_available is not False:
                    logger.warning(
                        "paddleocr not installed — PaddleOCR engine unavailable. "
                        "Install with: pip install paddleocr paddlepaddle"
                    )
                self._paddle_available = False
                return None
            except Exception as exc:
                logger.error("Failed to initialise PaddleOCR: %s", exc)
                self._paddle_available = False
                return None

        return self._ocr_instances[paddle_lang]

    @staticmethod
    def _ensure_3ch(image: np.ndarray) -> np.ndarray:
        """Convert grayscale to BGR 3-channel if needed."""
        import cv2
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    @staticmethod
    def _parse_results(results: list | None) -> list[WordResult]:
        """
        Parse PaddleOCR raw output into ``WordResult`` list.

        PaddleOCR returns a nested structure::

            [   # outer list = pages (we always pass 1 page)
                [   # inner list = detected text lines
                    [
                        [[x0,y0],[x1,y0],[x1,y1],[x0,y1]],  # bbox (4 corners)
                        (text, confidence)                    # (str, float 0-1)
                    ],
                    ...
                ]
            ]
        """
        if not results:
            return []

        words: list[WordResult] = []
        # Handle both single-page (list of lines) and multi-page formats
        page_results = results[0] if results and isinstance(results[0], list) else results

        if page_results is None:
            return []

        for item in page_results:
            if item is None or len(item) != 2:
                continue
            bbox_corners, text_conf = item
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) != 2:
                continue

            text, conf = text_conf
            if not isinstance(text, str) or not text.strip():
                continue

            # Convert 4-corner bbox to axis-aligned bbox
            try:
                xs = [p[0] for p in bbox_corners]
                ys = [p[1] for p in bbox_corners]
                x0, y0 = min(xs), min(ys)
                x1, y1 = max(xs), max(ys)
            except (TypeError, IndexError):
                x0 = y0 = x1 = y1 = 0.0

            words.append(WordResult(
                text=text,
                confidence=float(conf) * 100.0,  # normalise to 0–100
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
            ))

        return words
