"""
modules/htr/engines/base.py
────────────────────────────
Abstract HTR engine interface.

All concrete HTR engines (TrOCR, Tesseract Devanagari, …) implement this
protocol so the HandwritingReader can swap engines via configuration without
changing any other pipeline code.

Design-specified engines (Section 4.2 tech stack):
  Primary  : microsoft/trocr-base-handwritten  → TrOCREngine
  Secondary: Tesseract 5 (Devanagari / Hindi)   → TesseractDevanagariEngine
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ─── Region-level result from a single engine call ───────────────────────────

@dataclass
class RegionRecognition:
    """
    Raw recognition result for a single handwritten image crop.

    Confidence is in 0–100 range. For TrOCR this is computed from the
    geometric mean of per-token softmax probabilities × 100.
    For Tesseract it is the mean word confidence (already 0–100).
    """

    text: str
    confidence: float       # 0.0 – 100.0
    language: str = "en"    # ISO 639-1 of recognised text

    def is_empty(self) -> bool:
        """Return True if the recognition produced no meaningful text."""
        return not self.text.strip()


# ─── Abstract engine ──────────────────────────────────────────────────────────

class HTREngine(ABC):
    """
    Abstract base class for all HTR engine adapters.

    Subclasses must implement :meth:`recognize` and :meth:`engine_name`.
    They may override :meth:`warmup` for lazy model loading.
    """

    _warmed_up: bool = False

    def warmup(self) -> None:
        """
        Pre-load models. Called once before the first :meth:`recognize` call.
        Safe to call multiple times (idempotent).
        """

    @abstractmethod
    def recognize(self, image: np.ndarray, lang: str = "en") -> RegionRecognition:
        """
        Perform HTR on a single handwritten region crop.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale uint8 image of the handwritten region.
            The image should already be pre-processed (clean, deskewed).
        lang : str
            ISO 639-1 language hint. Engines may use this to select
            language-specific models (e.g. ``"hi"`` → Tesseract Devanagari).

        Returns
        -------
        RegionRecognition
            Recognised text, confidence (0–100), and detected language.
        """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Short identifier used in ``HTRRegion.htr_engine``."""

    @property
    def supported_languages(self) -> list[str]:
        """ISO 639-1 codes this engine can handle. Override in subclasses."""
        return ["en"]

    def supports_lang(self, lang: str) -> bool:
        """Return True if this engine supports the given language code."""
        return lang in self.supported_languages
