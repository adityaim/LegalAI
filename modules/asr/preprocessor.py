"""
modules/asr/preprocessor.py
─────────────────────────────
Audio pre-processing for Module 3 (ASR).

Responsibilities:
  1. Load audio from any common format (WAV, MP3, M4A, FLAC, OGG, WEBM)
     via soundfile (fast) with pydub fallback.
  2. Resample to 16 000 Hz mono float32 — the format Whisper expects.
     Reference: openai/whisper — load_audio() uses ffmpeg + 16kHz mono.
     https://github.com/openai/whisper/blob/main/whisper/audio.py
  3. Normalise amplitude to prevent clipping artefacts.
  4. Return a float32 numpy array ready for the engine.

Design notes
─────────────
• We do NOT embed audio or store it to disk permanently.
• Temp-file cleanup is the caller's responsibility (handled in reader.py).
• If soundfile fails (e.g. MP3 without libsndfile codec), we fall back to
  pydub → WAV in-memory → soundfile. Pydub itself requires ffmpeg on PATH.
• Normalisation is RMS-based: target_rms / actual_rms capped at 10×.
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Target sample rate required by Whisper
TARGET_SR = 16_000

# Supported audio extensions
SUPPORTED_AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".opus"
}

# RMS normalisation target (–20 dBFS → 0.1 linear)
_TARGET_RMS = 0.1
_MAX_GAIN   = 10.0


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Load an audio file and return a float32 mono array at *target_sr* Hz.

    Tries soundfile first (fast, pure-Python), then pydub (requires ffmpeg).

    Parameters
    ----------
    path : str | Path
        Path to the audio file.
    target_sr : int
        Target sample rate in Hz (default 16 000).

    Returns
    -------
    np.ndarray
        float32 array, shape (n_samples,), values in [-1, 1].

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not in SUPPORTED_AUDIO_EXTS.
    RuntimeError
        If the audio cannot be loaded by any available backend.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_AUDIO_EXTS)}"
        )

    # ── Attempt 1: soundfile (fast, handles WAV/FLAC/OGG natively) ──────────
    sf_err: Exception | None = None
    pd_err: Exception | None = None
    try:
        audio = _load_via_soundfile(path, target_sr)
        logger.debug("Loaded %s via soundfile (%d samples)", path.name, len(audio))
        return audio
    except Exception as e:
        sf_err = e
        logger.debug("soundfile failed for %s: %s — trying pydub", path.name, sf_err)

    # ── Attempt 2: pydub (handles MP3/M4A/AAC via ffmpeg) ───────────────────
    try:
        audio = _load_via_pydub(path, target_sr)
        logger.debug("Loaded %s via pydub (%d samples)", path.name, len(audio))
        return audio
    except Exception as e:
        pd_err = e
        raise RuntimeError(
            f"Failed to load audio from '{path}'. "
            f"soundfile error: {sf_err}; pydub error: {pd_err}. "
            "Install pydub + ffmpeg for MP3/M4A support: pip install pydub"
        ) from pd_err


def normalise_audio(audio: np.ndarray, target_rms: float = _TARGET_RMS) -> np.ndarray:
    """
    RMS-normalise *audio* to *target_rms* amplitude.

    Gain is capped at _MAX_GAIN to avoid amplifying pure silence.
    Returns the audio unchanged if it is effectively silent (rms < 1e-6).
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-6:
        return audio
    gain = min(_MAX_GAIN, target_rms / rms)
    return (audio * gain).clip(-1.0, 1.0).astype(np.float32)


def get_duration_s(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    """Return audio duration in seconds."""
    return len(audio) / sr


# ─── Private helpers ──────────────────────────────────────────────────────────

def _load_via_soundfile(path: Path, target_sr: int) -> np.ndarray:
    """Load audio using soundfile + optional resampling via scipy."""
    import soundfile as sf  # type: ignore[import]

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Mix down to mono
    audio = data.mean(axis=1)
    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def _load_via_pydub(path: Path, target_sr: int) -> np.ndarray:
    """Load audio using pydub (requires ffmpeg) and convert to float32."""
    from pydub import AudioSegment  # type: ignore[import]

    seg = AudioSegment.from_file(str(path))
    seg = seg.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)

    raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
    audio = raw / 32768.0  # int16 → float32 in [-1, 1]
    return audio.astype(np.float32)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio array from orig_sr to target_sr.

    Uses scipy.signal.resample_poly (high quality). Falls back to linear
    interpolation if scipy is not available.
    """
    if orig_sr == target_sr:
        return audio
    try:
        from scipy.signal import resample_poly  # type: ignore[import]
        from math import gcd
        g = gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        return resample_poly(audio, up, down).astype(np.float32)
    except ImportError:
        logger.warning("scipy not installed — using linear interpolation for resampling")
        n_samples = int(len(audio) * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_samples),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
