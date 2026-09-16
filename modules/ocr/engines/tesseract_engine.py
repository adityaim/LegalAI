"""
modules/ocr/engines/tesseract_engine.py
─────────────────────────────────────────
Tesseract 5 engine adapter — optional fallback OCR engine.

Requires:
    pip install pytesseract
    # + Tesseract binary installed on the system (tesseract-ocr package)

Tesseract language codes differ from both ISO 639-1 and PaddleOCR:
    English → "eng"
    Hindi   → "hin"
Combined configs like "eng+hin" are supported by Tesseract natively.
"""

from __future__ import annotations

import logging
from statistics import mean

import numpy as np

from .base import OCREngine, WordResult

logger = logging.getLogger(__name__)


# ─── Language mapping ────────────────────────────────────────────────────────

# ISO 639-1 → Tesseract lang code
LANG_MAP: dict[str, str] = {
    "en": "eng",
    "hi": "hin",
    "mr": "mar",
    "bn": "ben",
    "te": "tel",
    "ta": "tam",
    "gu": "guj",
    "pa": "pan",
}

DEFAULT_TESS_LANG = "eng"


# ─── Engine ──────────────────────────────────────────────────────────────────

class TesseractEngine(OCREngine):
    """
    OCR engine adapter wrapping pytesseract / Tesseract 5 LSTM.

    This is the *secondary* engine.  Use it when:
      • PaddleOCR is unavailable
      • High-quality scans where Tesseract LSTM performs well
      • Multi-language combined configs ("eng+hin")

    Parameters
    ----------
    default_lang : str
        ISO 639-1 default language code.
    psm : int
        Tesseract Page Segmentation Mode (default 6 = assume uniform block).
    oem : int
        OCR Engine Mode (default 3 = LSTM + legacy, auto-select).
    """

    def __init__(
        self,
        default_lang: str = "en",
        psm: int = 6,
        oem: int = 3,
    ) -> None:
        self._default_lang = default_lang
        self._psm = psm
        self._oem = oem
        self._available: bool | None = None

    @property
    def name(self) -> str:
        return "tesseract"

    def warmup(self) -> None:
        """Check that Tesseract is importable and available on PATH."""
        self._check_available()

    def recognize(self, image: np.ndarray, lang: str = "en") -> list[WordResult]:
        """
        Run Tesseract on *image* and return word-level results.

        Uses ``image_to_data`` (HOCR) to obtain per-word bounding boxes and
        confidence scores, which Tesseract's ``image_to_string`` does not
        provide directly.
        """
        if not self._check_available():
            return []

        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            logger.warning("pytesseract not installed — Tesseract engine unavailable")
            return []

        tess_lang = self._iso_to_tess(lang)
        config = f"--oem {self._oem} --psm {self._psm}"

        # Tesseract works best with PIL Images
        img_pil = self._to_pil(image)

        try:
            data = pytesseract.image_to_data(
                img_pil,
                lang=tess_lang,
                config=config,
                output_type=Output.DICT,
            )
        except Exception as exc:
            logger.error("Tesseract image_to_data raised: %s", exc)
            return []

        return self._parse_data(data)

    # ── Private helpers ────────────────────────────────────────────────────

    def _check_available(self) -> bool:
        if self._available is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._available = True
            except Exception:
                logger.warning(
                    "Tesseract not available.  Install tesseract-ocr system package "
                    "and pytesseract Python package."
                )
                self._available = False
        return self._available

    def _iso_to_tess(self, iso: str) -> str:
        """Convert ISO 639-1 code (or '+'-joined combo) to Tesseract lang string."""
        parts = iso.split("+")
        tess_parts = [LANG_MAP.get(p.strip(), DEFAULT_TESS_LANG) for p in parts]
        return "+".join(tess_parts)

    @staticmethod
    def _to_pil(image: np.ndarray):  # noqa: ANN205
        """Convert numpy array to PIL Image for pytesseract."""
        from PIL import Image
        if image.ndim == 2:
            return Image.fromarray(image, mode="L")
        import cv2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _parse_data(data: dict) -> list[WordResult]:
        """
        Parse pytesseract ``image_to_data`` output dict into ``WordResult`` list.

        Tesseract returns confidence as -1 for non-word rows (e.g. block/line
        separators) — we filter those out.
        """
        words: list[WordResult] = []
        n = len(data.get("text", []))

        for i in range(n):
            text: str = data["text"][i]
            conf: int = data["conf"][i]

            if not isinstance(text, str) or not text.strip():
                continue
            if conf < 0:
                continue  # Tesseract uses -1 for structure rows

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            words.append(WordResult(
                text=text,
                confidence=float(conf),  # Tesseract already returns 0–100
                x0=float(x),
                y0=float(y),
                x1=float(x + w),
                y1=float(y + h),
            ))

        return words
