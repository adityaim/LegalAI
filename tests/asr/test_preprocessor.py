"""
tests/asr/test_preprocessor.py
────────────────────────────────
Unit tests for the ASR audio pre-processor (Module 3).

Tests cover:
  • normalise_audio() — RMS normalisation, gain capping, silence handling
  • get_duration_s()  — duration calculation
  • load_audio()      — FileNotFoundError, ValueError for unsupported extensions
  • SUPPORTED_AUDIO_EXTS set contents
  • _resample()       — fallback interpolation (scipy-free)

Note: actual audio file I/O is tested via mocking to avoid requiring
soundfile/pydub in the CI environment.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modules.asr.preprocessor import (
    SUPPORTED_AUDIO_EXTS,
    TARGET_SR,
    get_duration_s,
    normalise_audio,
    _resample,
)


# ─── normalise_audio ──────────────────────────────────────────────────────────

class TestNormaliseAudio:
    def test_already_normalised(self):
        rng = np.random.default_rng(42)
        audio = rng.uniform(-0.1, 0.1, 16_000).astype(np.float32)
        out = normalise_audio(audio, target_rms=0.1)
        assert out.dtype == np.float32
        rms = float(np.sqrt(np.mean(out ** 2)))
        assert abs(rms - 0.1) < 0.01

    def test_quiet_audio_amplified(self):
        audio = np.full(16_000, 0.001, dtype=np.float32)
        out = normalise_audio(audio, target_rms=0.1)
        rms = float(np.sqrt(np.mean(out ** 2)))
        assert rms > 0.001  # must have been amplified

    def test_gain_capped_at_10x(self):
        # Near-silence: original RMS ≈ 1e-8, gain should be capped at 10
        audio = np.full(16_000, 1e-8, dtype=np.float32)
        out = normalise_audio(audio)
        assert np.all(np.abs(out) <= 1.0)  # no clipping beyond [-1, 1]

    def test_pure_silence_unchanged(self):
        audio = np.zeros(16_000, dtype=np.float32)
        out = normalise_audio(audio)
        assert np.all(out == 0.0)

    def test_output_clipped_to_unity(self):
        audio = np.full(16_000, 10.0, dtype=np.float32)
        out = normalise_audio(audio)
        assert float(np.max(np.abs(out))) <= 1.0

    def test_output_is_float32(self):
        audio = np.zeros(16_000, dtype=np.float64)
        out = normalise_audio(audio.astype(np.float32))
        assert out.dtype == np.float32


# ─── get_duration_s ───────────────────────────────────────────────────────────

class TestGetDurationS:
    def test_one_second(self):
        audio = np.zeros(TARGET_SR, dtype=np.float32)
        assert get_duration_s(audio) == pytest.approx(1.0)

    def test_half_second(self):
        audio = np.zeros(TARGET_SR // 2, dtype=np.float32)
        assert get_duration_s(audio) == pytest.approx(0.5)

    def test_custom_sr(self):
        audio = np.zeros(8_000, dtype=np.float32)
        assert get_duration_s(audio, sr=8_000) == pytest.approx(1.0)

    def test_empty_audio(self):
        assert get_duration_s(np.array([], dtype=np.float32)) == pytest.approx(0.0)


# ─── _resample ───────────────────────────────────────────────────────────────

class TestResample:
    def test_same_rate_no_op(self):
        audio = np.ones(16_000, dtype=np.float32)
        out = _resample(audio, 16_000, 16_000)
        assert len(out) == 16_000

    def test_upsample_length(self):
        # 8kHz → 16kHz should double the length
        audio = np.ones(8_000, dtype=np.float32)
        out = _resample(audio, 8_000, 16_000)
        assert len(out) == pytest.approx(16_000, abs=5)

    def test_downsample_length(self):
        # 32kHz → 16kHz should halve the length
        audio = np.ones(32_000, dtype=np.float32)
        out = _resample(audio, 32_000, 16_000)
        assert len(out) == pytest.approx(16_000, abs=5)

    def test_output_dtype_float32(self):
        audio = np.ones(8_000, dtype=np.float32)
        out = _resample(audio, 8_000, 16_000)
        assert out.dtype == np.float32

    def test_fallback_interpolation(self):
        """Test linear interpolation fallback when scipy is unavailable."""
        audio = np.linspace(0, 1, 8_000, dtype=np.float32)
        with patch.dict("sys.modules", {"scipy": None, "scipy.signal": None}):
            out = _resample(audio, 8_000, 16_000)
        assert len(out) == pytest.approx(16_000, abs=5)


# ─── SUPPORTED_AUDIO_EXTS ────────────────────────────────────────────────────

class TestSupportedExts:
    def test_wav_supported(self):
        assert ".wav" in SUPPORTED_AUDIO_EXTS

    def test_mp3_supported(self):
        assert ".mp3" in SUPPORTED_AUDIO_EXTS

    def test_m4a_supported(self):
        assert ".m4a" in SUPPORTED_AUDIO_EXTS

    def test_flac_supported(self):
        assert ".flac" in SUPPORTED_AUDIO_EXTS

    def test_ogg_supported(self):
        assert ".ogg" in SUPPORTED_AUDIO_EXTS

    def test_mp4_not_supported(self):
        assert ".mp4" not in SUPPORTED_AUDIO_EXTS


# ─── load_audio error handling ───────────────────────────────────────────────

class TestLoadAudioErrors:
    def test_file_not_found(self, tmp_path):
        from modules.asr.preprocessor import load_audio
        with pytest.raises(FileNotFoundError):
            load_audio(tmp_path / "nonexistent.wav")

    def test_unsupported_extension(self, tmp_path):
        from modules.asr.preprocessor import load_audio
        bad_file = tmp_path / "document.pdf"
        bad_file.write_bytes(b"fake content")
        with pytest.raises(ValueError, match="Unsupported audio format"):
            load_audio(bad_file)

    def test_both_backends_fail_raises_runtime(self, tmp_path):
        """If both soundfile and pydub fail, RuntimeError is raised."""
        from modules.asr.preprocessor import load_audio
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"not a real wav file")

        with patch("modules.asr.preprocessor._load_via_soundfile", side_effect=Exception("sf fail")), \
             patch("modules.asr.preprocessor._load_via_pydub", side_effect=Exception("pd fail")):
            with pytest.raises(RuntimeError, match="Failed to load audio"):
                load_audio(wav_file)
