"""
modules/ocr/engines/base.py
────────────────────────────
Abstract OCR engine interface.

All concrete engines (PaddleOCR, Tesseract, …) implement this protocol so
the reader can swap engines via configuration without changing any other code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ─── Word-level result ───────────────────────────────────────────────────────

@dataclass
class WordResult:
    """
    A single recognised word with its spatial location and confidence.

    Coordinates are in *page-pixel* space (top-left origin), matching the
    rasterised page image dimensions.
    """

    text: str
    confidence: float          # 0.0–100.0
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    def is_valid(self) -> bool:
        """Return True if the word contains non-whitespace text."""
        return bool(self.text.strip())


# ─── Abstract engine ─────────────────────────────────────────────────────────

class OCREngine(ABC):
    """
    Abstract base class for all OCR engine adapters.

    Subclasses must implement :meth:`recognize` and :meth:`name`.
    They may also override :meth:`warmup` for lazy model loading.
    """

    _warmed_up: bool = False

    # ── Public API ─────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Pre-load models / compile graphs.  Called once before the first
        :meth:`recognize` call.  Safe to call multiple times (idempotent).
        """

    @abstractmethod
    def recognize(
        self, image: np.ndarray, lang: str = "en"
    ) -> list[WordResult]:
        """
        Perform OCR on *image* and return word-level results.

        Parameters
        ----------
        image : np.ndarray
            Pre-processed grayscale or binary uint8 image.
        lang : str
            ISO 639-1 language code hint (e.g. ``"en"``, ``"hi"``).

        Returns
        -------
        list[WordResult]
            Recognised words in approximate reading order.  The list may be
            empty for blank or unrecognisable images.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in ``OCRBlock.ocr_engine``."""

    # ── Convenience helpers ────────────────────────────────────────────────

    def words_to_text(self, words: Sequence[WordResult]) -> str:
        """Join word texts in order, inserting line breaks heuristically."""
        if not words:
            return ""
        lines: list[list[str]] = []
        current_line: list[WordResult] = [words[0]]
        LINE_GAP_THRESHOLD = 8  # px: new line if y-gap exceeds this

        for word in words[1:]:
            prev = current_line[-1]
            # If the word's top is significantly below the previous word's
            # bottom, start a new line.
            if word.y0 - prev.y1 > LINE_GAP_THRESHOLD:
                lines.append([w.text for w in current_line])
                current_line = [word]
            else:
                current_line.append(word)
        lines.append([w.text for w in current_line])

        return "\n".join(" ".join(line) for line in lines)

    def mean_confidence(self, words: Sequence[WordResult]) -> float:
        """Return the mean confidence of a list of WordResults (0–100)."""
        valid = [w.confidence for w in words if w.confidence >= 0]
        return float(np.mean(valid)) if valid else 0.0
