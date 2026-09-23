"""
modules/asr/reader.py
──────────────────────
ASRReader — the main pipeline orchestrator for Module 3 (ASR Voice-to-Text).

Pipeline overview::

    process(path | bytes | np.ndarray)
    ├── load_audio()                  # preprocessor: load + resample → 16kHz float32
    ├── normalise_audio()             # RMS normalisation
    ├── engine.transcribe()           # WhisperEngine or FasterWhisperEngine
    ├── postprocessor.process_segments()   # logprob→confidence, silence drop, corrections
    ├── postprocessor.assemble_transcript()
    ├── postprocessor.detect_intent()
    └── assemble ASRResult()

Engine selection strategy
──────────────────────────
  Default (engine=None):
    → WhisperEngine("base") on CPU — no GPU required.
  engine_backend="faster_whisper":
    → FasterWhisperEngine("base", compute_type="int8") — 2–4× faster on CPU.
  Inject any ASREngine subclass for testing or custom backends.

Language handling
──────────────────
  language=None (default) → Whisper auto-detects from first 30s of audio.
  language="hi"           → forces Hindi; improves accuracy for Hindi legal audio.
  language="en"           → forces English.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .engines.base import ASREngine, TranscriptionResult
from .engines.whisper_engine import WhisperEngine
from .engines.faster_whisper_engine import FasterWhisperEngine
from .postprocessor import ASRPostprocessor
from .preprocessor import SUPPORTED_AUDIO_EXTS, get_duration_s, load_audio, normalise_audio
from .schema import ASRResult, ASRSegment

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ReaderConfig:
    """Configuration for ASRReader."""

    # Primary language (ISO 639-1); None → Whisper auto-detects
    language: str | None = None

    # Engine backend: "whisper" (default) | "faster_whisper"
    engine_backend: Literal["whisper", "faster_whisper"] = "whisper"

    # Whisper model size: tiny | base | small | medium | large
    model_size: str = "base"

    # Compute device: "cpu" | "cuda"
    device: str = "cpu"

    # faster-whisper only: quantisation type
    compute_type: str = "int8"

    # Enable legal term corrections in post-processor
    enable_legal_corrections: bool = True

    # Silence threshold: segments with no_speech_prob above this are dropped
    no_speech_threshold: float = 0.6

    # Optional initial prompt to bias Whisper decoder
    initial_prompt: str | None = None


# ─── Main reader ─────────────────────────────────────────────────────────────

class ASRReader:
    """
    Main ASR pipeline. Accepts audio file paths or in-memory numpy arrays.

    Usage::

        reader = ASRReader()
        result = reader.process("court_hearing.mp3")
        print(result.transcript)
        print(result.language)
        print(result.intent)
        print(result.to_fusion_text())   # for Module 4

        # Force Hindi + faster engine
        config = ReaderConfig(language="hi", engine_backend="faster_whisper")
        reader = ASRReader(config=config)
        result = reader.process("hindi_deposition.wav")
    """

    def __init__(
        self,
        config: ReaderConfig | None = None,
        engine: ASREngine | None = None,
    ) -> None:
        self.config = config or ReaderConfig()

        # Injected engine (used in tests) or build from config
        self._engine: ASREngine = engine if engine is not None else self._build_engine()

        # Post-processor
        self._postprocessor = ASRPostprocessor(
            no_speech_threshold=self.config.no_speech_threshold,
            enable_legal_corrections=self.config.enable_legal_corrections,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, source: "str | Path | np.ndarray") -> ASRResult:
        """
        Transcribe audio from *source*.

        Parameters
        ----------
        source : str | Path | np.ndarray
            Audio file path (WAV, MP3, M4A, FLAC, OGG, WEBM, AAC) or
            a float32 numpy array already at 16 000 Hz mono.

        Returns
        -------
        ASRResult
            Full transcript, time-aligned segments, detected language,
            legal intent, and per-segment confidence scores.
        """
        start = time.perf_counter()
        doc_id = str(uuid.uuid4())

        # ── Load audio ─────────────────────────────────────────────────────
        if isinstance(source, np.ndarray):
            audio = source.astype(np.float32)
            logger.info("ASR: processing in-memory audio array (%d samples)", len(audio))
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            ext = path.suffix.lower()
            if ext not in SUPPORTED_AUDIO_EXTS:
                raise ValueError(
                    f"Unsupported audio format '{ext}'. "
                    f"Supported: {sorted(SUPPORTED_AUDIO_EXTS)}"
                )
            logger.info("ASR: loading %s", path.name)
            audio = load_audio(path)

        # ── Normalise ──────────────────────────────────────────────────────
        audio = normalise_audio(audio)
        duration_s = get_duration_s(audio)
        logger.info("ASR: audio duration=%.1fs", duration_s)

        # ── Transcribe ─────────────────────────────────────────────────────
        raw: TranscriptionResult = self._engine.transcribe(
            audio,
            language=self.config.language,
            initial_prompt=self.config.initial_prompt,
        )
        logger.info(
            "ASR engine '%s': %d raw segments, detected lang=%s (prob=%.2f)",
            self._engine.engine_name,
            len(raw.segments),
            raw.language,
            raw.language_probability,
        )

        # ── Post-process ───────────────────────────────────────────────────
        segments: list[ASRSegment] = self._postprocessor.process_segments(
            raw.segments,
            engine_tag=self._engine.engine_name,
        )
        transcript = self._postprocessor.assemble_transcript(segments)
        intent = self._postprocessor.detect_intent(transcript)
        low_conf_count = sum(1 for s in segments if s.review_required)

        elapsed = time.perf_counter() - start
        logger.info(
            "ASR complete: %d segments, %d flagged, intent=%s, %.2fs",
            len(segments), low_conf_count, intent, elapsed,
        )

        return ASRResult(
            document_id=doc_id,
            source="ASR",
            language=raw.language,
            language_probability=raw.language_probability,
            duration_s=duration_s,
            transcript=transcript,
            segments=segments,
            intent=intent,
            processing_time_s=round(elapsed, 3),
            low_confidence_segments=low_conf_count,
        )

    def warmup(self) -> None:
        """Pre-load the ASR engine model. Call once at server startup."""
        logger.info("ASR: warming up engine '%s'…", self._engine.engine_name)
        self._engine.warmup()
        logger.info("ASR: engine ready.")

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_engine(self) -> ASREngine:
        """Instantiate the configured ASR engine."""
        if self.config.engine_backend == "faster_whisper":
            return FasterWhisperEngine(
                model_size=self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                legal_prompt=self.config.initial_prompt,
            )
        # Default: openai/whisper
        return WhisperEngine(
            model_size=self.config.model_size,
            device=self.config.device,
            legal_prompt=self.config.initial_prompt,
        )
