"""
modules/asr/engines/base.py
────────────────────────────
Abstract ASR engine interface.

All concrete ASR engines (WhisperEngine, FasterWhisperEngine, …) implement
this protocol so ASRReader can swap engines via configuration without
changing any other pipeline code.

Design-specified engines (Section 4.3 tech stack):
  Primary  : openai/whisper (base model)   → WhisperEngine
  Secondary: SYSTRAN/faster-whisper        → FasterWhisperEngine (CPU-optimised)

Reference:
  faster-whisper Segment dataclass — avg_logprob, no_speech_prob, words:
  https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ─── Raw segment result from a single engine call ────────────────────────────

@dataclass
class SegmentResult:
    """
    Raw transcription result for a single time-aligned audio segment.

    avg_logprob mirrors Whisper's Segment.avg_logprob (in (-∞, 0]).
    The ASR postprocessor maps this to confidence 0–100.

    Reference:
      faster-whisper Segment:
        avg_logprob: float     — average log-probability over decoded tokens
        no_speech_prob: float  — probability of silence / no speech
    """
    segment_id: int
    start_s: float
    end_s: float
    text: str
    avg_logprob: float          # (-∞, 0]; 0 = perfect, -4 ≈ unusable
    no_speech_prob: float = 0.0 # [0, 1]
    language: str = "en"

    def is_empty(self) -> bool:
        """Return True if the segment produced no meaningful text."""
        return not self.text.strip()

    def is_silence(self, threshold: float = 0.6) -> bool:
        """Return True if Whisper considers this a silent segment."""
        return self.no_speech_prob > threshold


@dataclass
class TranscriptionResult:
    """
    Full engine-level output: list of SegmentResults + detected language.
    """
    segments: list[SegmentResult] = field(default_factory=list)
    language: str = "en"
    language_probability: float = 1.0
    duration_s: float = 0.0


# ─── Abstract engine ──────────────────────────────────────────────────────────

class ASREngine(ABC):
    """
    Abstract base class for all ASR engine adapters.

    Subclasses must implement :meth:`transcribe` and :meth:`engine_name`.
    They may override :meth:`warmup` for lazy model loading.
    """

    _warmed_up: bool = False

    def warmup(self) -> None:
        """
        Pre-load models. Called once before the first transcribe() call.
        Safe to call multiple times (idempotent).
        """

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe a float32 mono audio array (16 000 Hz).

        Parameters
        ----------
        audio : np.ndarray
            float32 array, shape (n_samples,), values in [-1, 1].
            Must already be at 16 000 Hz (handled by preprocessor).
        language : str | None
            ISO 639-1 language hint. If None, Whisper auto-detects.
        initial_prompt : str | None
            Optional prompt to bias the decoder towards legal vocabulary.

        Returns
        -------
        TranscriptionResult
            All segments with text, timestamps, avg_logprob, no_speech_prob.
        """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Short identifier used in ASRSegment.asr_engine."""

    @property
    def supported_languages(self) -> list[str]:
        """ISO 639-1 codes this engine can handle. Override in subclasses."""
        return ["en"]

    def supports_lang(self, lang: str) -> bool:
        """Return True if this engine supports the given language code."""
        return lang in self.supported_languages
