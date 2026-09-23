"""
modules/asr/engines/whisper_engine.py
───────────────────────────────────────
OpenAI Whisper engine adapter for Module 3 (ASR).

Primary ASR engine. Wraps openai/whisper to produce SegmentResult objects
compatible with the ASR pipeline contract.

Reference (openai/whisper):
  whisper.transcribe() returns:
    {
      "text": str,
      "segments": [
        {
          "id": int, "start": float, "end": float,
          "text": str, "avg_logprob": float,
          "no_speech_prob": float, "compression_ratio": float,
          "tokens": List[int], "temperature": float
        }, ...
      ],
      "language": str
    }
  Source: https://github.com/openai/whisper/blob/main/whisper/transcribe.py

Model sizes (smallest → largest):
  tiny   (~39 M params)  — fastest, lowest accuracy
  base   (~74 M params)  — recommended default for CPU
  small  (~244 M params)
  medium (~769 M params)
  large  (~1550 M params) — best accuracy, requires GPU

Legal domain prompt
────────────────────
Whisper's initial_prompt biases the decoder vocabulary. We default to a
short legal context prompt that improves recognition of Indian legal terms
like "IPC", "Section 138", "NI Act", "SC", "HC", etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import ASREngine, SegmentResult, TranscriptionResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Whisper model sizes available via openai/whisper
WHISPER_MODELS = {"tiny", "base", "small", "medium", "large",
                  "tiny.en", "base.en", "small.en", "medium.en",
                  "large-v1", "large-v2", "large-v3", "turbo"}

# Default legal-domain initial prompt (improves accuracy on legal transcripts)
DEFAULT_LEGAL_PROMPT = (
    "Legal proceedings transcript. Indian law. "
    "Section 138, IPC, CrPC, NI Act, Supreme Court, High Court, "
    "affidavit, plaintiff, defendant, jurisdiction."
)

# SAMPLE_RATE expected by Whisper
WHISPER_SR = 16_000


class WhisperEngine(ASREngine):
    """
    OpenAI Whisper engine adapter.

    Lazy-loads the model on the first transcribe() call to avoid import-time
    overhead. Falls back gracefully if the 'whisper' package is not installed.

    Usage::

        engine = WhisperEngine(model_size="base", device="cpu")
        result = engine.transcribe(audio_float32, language="en")
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        fp16: bool = False,
        legal_prompt: str | None = DEFAULT_LEGAL_PROMPT,
    ) -> None:
        if model_size not in WHISPER_MODELS:
            raise ValueError(
                f"Unknown Whisper model '{model_size}'. "
                f"Choose from: {sorted(WHISPER_MODELS)}"
            )
        self._model_size = model_size
        self._device = device
        self._fp16 = fp16 and device != "cpu"   # fp16 not supported on CPU
        self._legal_prompt = legal_prompt
        self._model = None

    # ── Public API ────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Pre-load Whisper model. Called once; idempotent."""
        if self._model is None:
            self._load_model()

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe float32 mono 16kHz audio using OpenAI Whisper.

        Parameters
        ----------
        audio : np.ndarray
            float32 mono array at 16 000 Hz.
        language : str | None
            ISO 639-1 code, or None for auto-detection.
        initial_prompt : str | None
            Prompt to prepend; defaults to DEFAULT_LEGAL_PROMPT.

        Returns
        -------
        TranscriptionResult
        """
        if self._model is None:
            self._load_model()

        prompt = initial_prompt if initial_prompt is not None else self._legal_prompt
        duration_s = len(audio) / WHISPER_SR

        # openai/whisper transcribe() call
        # Reference: whisper/transcribe.py — transcribe(model, audio, ...)
        raw = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt,
            fp16=self._fp16,
            verbose=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # fallback temperatures
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            word_timestamps=False,
        )

        detected_lang = raw.get("language", language or "en")
        lang_prob = 1.0   # openai/whisper does not expose language probability

        segments: list[SegmentResult] = []
        for seg in raw.get("segments", []):
            segments.append(SegmentResult(
                segment_id=seg["id"],
                start_s=float(seg["start"]),
                end_s=float(seg["end"]),
                text=seg["text"].strip(),
                avg_logprob=float(seg.get("avg_logprob", -0.5)),
                no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                language=detected_lang,
            ))

        return TranscriptionResult(
            segments=segments,
            language=detected_lang,
            language_probability=lang_prob,
            duration_s=duration_s,
        )

    @property
    def engine_name(self) -> str:
        return "whisper"

    @property
    def supported_languages(self) -> list[str]:
        """Whisper supports 99 languages; we surface the primary ones."""
        return [
            "en", "hi", "mr", "pa", "bn", "ur", "ta", "te", "kn", "ml",
            "gu", "or", "as", "sa", "ne", "si",
            "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar",
        ]

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load openai/whisper model. Raises ImportError if not installed."""
        try:
            import whisper  # type: ignore[import]
            logger.info("Loading Whisper model '%s' on %s…", self._model_size, self._device)
            self._model = whisper.load_model(self._model_size, device=self._device)
            logger.info("Whisper model '%s' loaded successfully.", self._model_size)
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is not installed. "
                "Install with: pip install openai-whisper"
            ) from exc
