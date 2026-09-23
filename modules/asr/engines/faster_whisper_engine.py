"""
modules/asr/engines/faster_whisper_engine.py
─────────────────────────────────────────────
SYSTRAN faster-whisper engine adapter for Module 3 (ASR).

Optional CPU-optimised alternative to WhisperEngine. faster-whisper uses
CTranslate2 for inference — typically 2–4× faster than openai/whisper on CPU
with similar accuracy.

Reference (SYSTRAN/faster-whisper):
  WhisperModel.transcribe() returns (segments_generator, TranscriptionInfo):
    info.language              — detected language (ISO 639-1)
    info.language_probability  — confidence of language detection [0, 1]
    info.duration              — total audio duration in seconds
    segment.id                 — 0-indexed segment counter
    segment.start              — start time in seconds
    segment.end                — end time in seconds
    segment.text               — transcribed text
    segment.avg_logprob        — average log-probability (−∞, 0]
    segment.no_speech_prob     — probability of silence [0, 1]
  Source: https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py

Quantisation options (model_size_or_path="base", compute_type=...):
  "int8"        — smallest, fastest, ~1% accuracy drop — recommended for CPU
  "int8_float16"— GPU mixed precision
  "float16"     — GPU full half precision
  "float32"     — full precision (same as openai/whisper on CPU)

Legal domain prompt: same as WhisperEngine for consistency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import ASREngine, SegmentResult, TranscriptionResult
from .whisper_engine import DEFAULT_LEGAL_PROMPT, WHISPER_SR

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FasterWhisperEngine(ASREngine):
    """
    SYSTRAN faster-whisper engine adapter.

    Lazy-loads the CTranslate2 model on the first transcribe() call.
    Falls back gracefully if the 'faster-whisper' package is not installed.

    Usage::

        engine = FasterWhisperEngine(model_size="base", compute_type="int8")
        result = engine.transcribe(audio_float32, language="hi")
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",     # "int8" recommended for CPU
        legal_prompt: str | None = DEFAULT_LEGAL_PROMPT,
        num_workers: int = 1,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._legal_prompt = legal_prompt
        self._num_workers = num_workers
        self._model = None

    # ── Public API ────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Pre-load faster-whisper model. Called once; idempotent."""
        if self._model is None:
            self._load_model()

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe float32 mono 16kHz audio using faster-whisper.

        Returns
        -------
        TranscriptionResult
        """
        if self._model is None:
            self._load_model()

        prompt = initial_prompt if initial_prompt is not None else self._legal_prompt
        duration_s = len(audio) / WHISPER_SR

        # faster-whisper WhisperModel.transcribe()
        # Reference: faster_whisper/transcribe.py — transcribe(audio, ...)
        # Returns: (generator[Segment], TranscriptionInfo)
        raw_segments, info = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt,
            beam_size=5,
            best_of=5,
            patience=1.0,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            word_timestamps=False,
            vad_filter=True,          # skip silence regions automatically
        )

        detected_lang = info.language
        lang_prob = float(info.language_probability)

        segments: list[SegmentResult] = []
        for seg in raw_segments:  # generator — consume fully
            segments.append(SegmentResult(
                segment_id=seg.id,
                start_s=float(seg.start),
                end_s=float(seg.end),
                text=seg.text.strip(),
                avg_logprob=float(seg.avg_logprob),
                no_speech_prob=float(seg.no_speech_prob),
                language=detected_lang,
            ))

        return TranscriptionResult(
            segments=segments,
            language=detected_lang,
            language_probability=lang_prob,
            duration_s=float(info.duration),
        )

    @property
    def engine_name(self) -> str:
        return "faster_whisper"

    @property
    def supported_languages(self) -> list[str]:
        """Supports all 99 Whisper languages."""
        return [
            "en", "hi", "mr", "pa", "bn", "ur", "ta", "te", "kn", "ml",
            "gu", "or", "as", "sa", "ne", "si",
            "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar",
        ]

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load faster-whisper model."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]
            logger.info(
                "Loading faster-whisper model '%s' on %s (compute_type=%s)…",
                self._model_size, self._device, self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                num_workers=self._num_workers,
            )
            logger.info("faster-whisper model '%s' loaded.", self._model_size)
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            ) from exc
