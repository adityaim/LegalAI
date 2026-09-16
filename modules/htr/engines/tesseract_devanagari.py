"""
modules/htr/engines/tesseract_devanagari.py
─────────────────────────────────────────────
Tesseract 5 engine for Hindi / Devanagari handwriting recognition.

Design spec (Section 4.2 tech stack):
  "Indian language HTR: Tesseract 5 (Devanagari) / Bhashini OCR API"

This engine activates when lang="hi" (or any Devanagari-script language)
is requested. It uses Tesseract's LSTM mode with the 'hin' language pack,
which is trained on Devanagari script and performs reasonably on neat
handwriting.

Limitations
────────────
Tesseract is NOT purpose-built for handwriting — it works best on
relatively neat, upright Devanagari script. Highly cursive or degraded
handwriting should be routed to a dedicated Devanagari HTR model
(e.g. Bhashini OCR API) once available.

Language support
─────────────────
  hi  → hin  (Hindi)
  mr  → mar  (Marathi, Devanagari script)
  ne  → nep  (Nepali, Devanagari script)
  sa  → san  (Sanskrit)

For other Indic scripts (Tamil, Telugu, etc.) Tesseract has separate
language packs; these can be added to LANG_MAP as needed.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import HTREngine, RegionRecognition

logger = logging.getLogger(__name__)

# ─── Language mapping ────────────────────────────────────────────────────────

LANG_MAP: dict[str, str] = {
    "hi": "hin",   # Hindi (Devanagari)
    "mr": "mar",   # Marathi (Devanagari)
    "ne": "nep",   # Nepali (Devanagari)
    "sa": "san",   # Sanskrit (Devanagari)
    # Non-Devanagari Indic scripts — add Tesseract lang packs as needed
    "ta": "tam",
    "te": "tel",
    "bn": "ben",
    "gu": "guj",
    "pa": "pan",
}

DEFAULT_TESS_LANG = "hin"

# Tesseract config optimised for handwriting (sparse text + LSTM)
# psm 6 = assume uniform block of text
# psm 11 = sparse text (better for notes/annotations)
TESS_CONFIG = "--oem 1 --psm 6"   # LSTM only, uniform block


class TesseractDevanagariEngine(HTREngine):
    """
    Tesseract 5 LSTM engine for Devanagari / Indic script handwriting.

    Parameters
    ----------
    default_lang : str
        ISO 639-1 code for the default language (default: ``"hi"``).
    psm : int
        Page segmentation mode (6=uniform block, 11=sparse text).
    """

    def __init__(self, default_lang: str = "hi", psm: int = 6) -> None:
        self._default_lang = default_lang
        self._psm = psm
        self._available: bool | None = None

    @property
    def engine_name(self) -> str:
        return "tesseract_devanagari"

    @property
    def supported_languages(self) -> list[str]:
        return list(LANG_MAP.keys())

    def warmup(self) -> None:
        """Check Tesseract availability."""
        self._check_available()

    def recognize(self, image: np.ndarray, lang: str = "hi") -> RegionRecognition:
        """
        Run Tesseract with the appropriate Indic language pack.

        Returns an empty RegionRecognition if Tesseract is not available.
        """
        if not self._check_available():
            return RegionRecognition(text="", confidence=0.0, language=lang)

        try:
            import pytesseract
            from pytesseract import Output

            tess_lang = LANG_MAP.get(lang, DEFAULT_TESS_LANG)
            config = f"--oem 1 --psm {self._psm}"

            pil_img = self._to_pil(image)

            # Get full data for confidence scores
            data = pytesseract.image_to_data(
                pil_img, lang=tess_lang, config=config, output_type=Output.DICT
            )
            text = pytesseract.image_to_string(
                pil_img, lang=tess_lang, config=config
            ).strip()

            confidence = self._extract_confidence(data)

        except Exception as exc:
            logger.error("Tesseract Devanagari error: %s", exc)
            return RegionRecognition(text="", confidence=0.0, language=lang)

        return RegionRecognition(text=text, confidence=confidence, language=lang)

    # ── Private helpers ────────────────────────────────────────────────────

    def _check_available(self) -> bool:
        if self._available is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._available = True
            except Exception:
                logger.warning(
                    "Tesseract not found. Install tesseract-ocr with hin language pack: "
                    "apt install tesseract-ocr-hin (Linux) or via installer on Windows."
                )
                self._available = False
        return self._available

    @staticmethod
    def _to_pil(image: np.ndarray):  # noqa: ANN205
        from PIL import Image
        import cv2
        if image.ndim == 2:
            return Image.fromarray(image, mode="L")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _extract_confidence(data: dict) -> float:
        """Extract mean confidence from Tesseract image_to_data output."""
        confs = [
            c for c in data.get("conf", [])
            if isinstance(c, (int, float)) and c >= 0
        ]
        if not confs:
            return 0.0
        return round(sum(confs) / len(confs), 2)
