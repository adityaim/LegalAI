"""
tests/asr/test_reader.py
─────────────────────────
Integration tests for ASRReader (Module 3).

Tests cover:
  • process() with a mock engine — full pipeline from numpy array
  • process() from a synthetic WAV file (no Whisper install required)
  • FileNotFoundError for missing files
  • ValueError for unsupported extension
  • Engine injection (dependency injection pattern)
  • warmup() is forwarded to the engine
  • ReaderConfig defaults and overrides
  • language auto-detection vs forced language
  • Low-confidence segment counting in result
  • to_fusion_text() contract with the reader
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import struct
import wave

import numpy as np
import pytest

from modules.asr.engines.base import ASREngine, SegmentResult, TranscriptionResult
from modules.asr.reader import ASRReader, ReaderConfig
from modules.asr.schema import ASRResult


# ─── Mock engine factory ──────────────────────────────────────────────────────

def make_mock_engine(
    segments: list[SegmentResult] | None = None,
    language: str = "en",
    lang_prob: float = 0.99,
    duration_s: float = 10.0,
) -> ASREngine:
    """Create a mock ASR engine that returns predictable TranscriptionResult."""
    if segments is None:
        segments = [
            SegmentResult(
                segment_id=0,
                start_s=0.0,
                end_s=5.0,
                text="The plaintiff filed a case under Section 138 of the NI Act.",
                avg_logprob=-0.4,
                no_speech_prob=0.02,
                language=language,
            ),
        ]

    engine = MagicMock(spec=ASREngine)
    engine.engine_name = "mock"
    engine.transcribe.return_value = TranscriptionResult(
        segments=segments,
        language=language,
        language_probability=lang_prob,
        duration_s=duration_s,
    )
    engine.warmup.return_value = None
    return engine


def make_wav_file(tmp_path: Path, duration_s: float = 1.0, sr: int = 16_000) -> Path:
    """Create a synthetic mono WAV file with silence."""
    wav_path = tmp_path / "test_audio.wav"
    n_samples = int(duration_s * sr)
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return wav_path


# ─── ReaderConfig tests ───────────────────────────────────────────────────────

class TestReaderConfig:
    def test_defaults(self):
        cfg = ReaderConfig()
        assert cfg.language is None
        assert cfg.engine_backend == "whisper"
        assert cfg.model_size == "base"
        assert cfg.device == "cpu"
        assert cfg.enable_legal_corrections is True
        assert cfg.no_speech_threshold == pytest.approx(0.6)

    def test_custom_config(self):
        cfg = ReaderConfig(language="hi", engine_backend="faster_whisper", model_size="small")
        assert cfg.language == "hi"
        assert cfg.engine_backend == "faster_whisper"
        assert cfg.model_size == "small"


# ─── ASRReader process() with mock engine ────────────────────────────────────

class TestASRReaderWithMockEngine:
    def test_process_numpy_array(self):
        engine = make_mock_engine()
        reader = ASRReader(engine=engine)
        audio = np.zeros(16_000, dtype=np.float32)
        result = reader.process(audio)

        assert isinstance(result, ASRResult)
        assert result.source == "ASR"
        assert result.language == "en"
        assert len(result.document_id) == 36  # UUID4

    def test_process_returns_transcript(self):
        engine = make_mock_engine()
        reader = ASRReader(engine=engine)
        audio = np.zeros(16_000, dtype=np.float32)
        result = reader.process(audio)
        assert "Section 138" in result.transcript

    def test_process_returns_segments(self):
        engine = make_mock_engine()
        reader = ASRReader(engine=engine)
        result = reader.process(np.zeros(16_000, dtype=np.float32))
        assert len(result.segments) >= 1

    def test_process_unique_document_ids(self):
        engine = make_mock_engine()
        reader = ASRReader(engine=engine)
        audio = np.zeros(16_000, dtype=np.float32)
        r1 = reader.process(audio)
        r2 = reader.process(audio)
        assert r1.document_id != r2.document_id

    def test_process_language_from_engine(self):
        engine = make_mock_engine(language="hi", lang_prob=0.97)
        reader = ASRReader(engine=engine)
        result = reader.process(np.zeros(16_000, dtype=np.float32))
        assert result.language == "hi"
        assert result.language_probability == pytest.approx(0.97)

    def test_process_intent_detected(self):
        segments = [SegmentResult(
            segment_id=0, start_s=0.0, end_s=5.0,
            text="Draft a legal notice for the respondent.",
            avg_logprob=-0.3, no_speech_prob=0.02, language="en",
        )]
        engine = make_mock_engine(segments=segments)
        reader = ASRReader(engine=engine)
        result = reader.process(np.zeros(16_000, dtype=np.float32))
        assert result.intent == "instruction"

    def test_process_question_intent(self):
        segments = [SegmentResult(
            segment_id=0, start_s=0.0, end_s=4.0,
            text="What does the NI Act say about dishonoured cheques?",
            avg_logprob=-0.3, no_speech_prob=0.02, language="en",
        )]
        engine = make_mock_engine(segments=segments)
        reader = ASRReader(engine=engine)
        result = reader.process(np.zeros(16_000, dtype=np.float32))
        assert result.intent == "question"

    def test_low_confidence_segments_counted(self):
        segments = [
            SegmentResult(
                segment_id=0, start_s=0.0, end_s=5.0,
                text="Good segment.", avg_logprob=-0.3, no_speech_prob=0.02, language="en",
            ),
            SegmentResult(
                segment_id=1, start_s=5.0, end_s=10.0,
                text="Bad segment.", avg_logprob=-3.9, no_speech_prob=0.05, language="en",
            ),
        ]
        engine = make_mock_engine(segments=segments)
        reader = ASRReader(engine=engine)
        result = reader.process(np.zeros(16_000, dtype=np.float32))
        assert result.low_confidence_segments == 1

    def test_silence_segments_dropped(self):
        segments = [
            SegmentResult(
                segment_id=0, start_s=0.0, end_s=5.0,
                text="Real speech.", avg_logprob=-0.3, no_speech_prob=0.02, language="en",
            ),
            SegmentResult(
                segment_id=1, start_s=5.0, end_s=10.0,
                text="Silence segment.", avg_logprob=-0.5, no_speech_prob=0.95, language="en",
            ),
        ]
        engine = make_mock_engine(segments=segments)
        reader = ASRReader(engine=engine)
        result = reader.process(np.zeros(16_000, dtype=np.float32))
        assert len(result.segments) == 1

    def test_to_fusion_text_from_result(self):
        engine = make_mock_engine()
        reader = ASRReader(engine=engine)
        result = reader.process(np.zeros(16_000, dtype=np.float32))
        ft = result.to_fusion_text()
        assert isinstance(ft, str)
        assert len(ft) > 0

    def test_processing_time_is_non_negative(self):
        engine = make_mock_engine()
        reader = ASRReader(engine=engine)
        result = reader.process(np.zeros(16_000, dtype=np.float32))
        assert result.processing_time_s >= 0.0

    def test_warmup_forwarded(self):
        engine = make_mock_engine()
        reader = ASRReader(engine=engine)
        reader.warmup()
        engine.warmup.assert_called_once()

    def test_engine_transcribe_called_with_language(self):
        engine = make_mock_engine()
        config = ReaderConfig(language="hi")
        reader = ASRReader(config=config, engine=engine)
        reader.process(np.zeros(16_000, dtype=np.float32))
        call_kwargs = engine.transcribe.call_args
        assert call_kwargs.kwargs.get("language") == "hi" or \
               (call_kwargs.args and call_kwargs.args[1] == "hi")


# ─── ASRReader process() from file ───────────────────────────────────────────

class TestASRReaderFromFile:
    def test_file_not_found(self, tmp_path):
        engine = make_mock_engine()
        reader = ASRReader(engine=engine)
        with pytest.raises(FileNotFoundError):
            reader.process(tmp_path / "nonexistent.wav")

    def test_unsupported_extension(self, tmp_path):
        engine = make_mock_engine()
        reader = ASRReader(engine=engine)
        bad_file = tmp_path / "document.xyz"
        bad_file.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported audio format"):
            reader.process(bad_file)

    def test_wav_file_processed(self, tmp_path):
        """End-to-end with a real WAV file + mock engine (no Whisper needed)."""
        wav = make_wav_file(tmp_path, duration_s=1.0)
        engine = make_mock_engine()

        # Patch load_audio to return our synthetic audio directly
        with patch("modules.asr.reader.load_audio", return_value=np.zeros(16_000, dtype=np.float32)):
            reader = ASRReader(engine=engine)
            result = reader.process(wav)

        assert isinstance(result, ASRResult)
        assert result.source == "ASR"

    def test_result_serialisation_from_file(self, tmp_path):
        wav = make_wav_file(tmp_path, duration_s=1.0)
        engine = make_mock_engine()
        with patch("modules.asr.reader.load_audio", return_value=np.zeros(16_000, dtype=np.float32)):
            reader = ASRReader(engine=engine)
            result = reader.process(wav)

        data = result.model_dump()
        restored = ASRResult.model_validate(data)
        assert restored.document_id == result.document_id


# ─── Engine building ─────────────────────────────────────────────────────────

class TestEngineBuild:
    def test_whisper_engine_built_by_default(self):
        from modules.asr.engines.whisper_engine import WhisperEngine
        config = ReaderConfig(engine_backend="whisper", model_size="base")
        reader = ASRReader(config=config, engine=make_mock_engine())  # injected
        # Build via config (no injection)
        with patch("modules.asr.engines.whisper_engine.WhisperEngine._load_model"):
            reader2 = ASRReader(config=config)
        assert isinstance(reader2._engine, WhisperEngine)

    def test_faster_whisper_engine_built_when_configured(self):
        from modules.asr.engines.faster_whisper_engine import FasterWhisperEngine
        config = ReaderConfig(engine_backend="faster_whisper", model_size="base")
        with patch("modules.asr.engines.faster_whisper_engine.FasterWhisperEngine._load_model"):
            reader = ASRReader(config=config)
        assert isinstance(reader._engine, FasterWhisperEngine)
