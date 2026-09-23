"""
api/asr_router.py
──────────────────
FastAPI router for Module 3: ASR Voice-to-Text.

Endpoints
─────────
POST /api/v1/asr/process
    Upload an audio file; returns ASRResult JSON with full transcript,
    time-aligned segments, confidence scores, detected language, and legal intent.

GET  /api/v1/asr/health
    Check engine availability (whisper, faster-whisper, soundfile, pydub).

Design notes
─────────────
• Temp files are always deleted after processing (NFR-8 compliance).
• One ASRReader is instantiated per request; for high-throughput production,
  use a dependency-injected shared reader pre-warmed at startup.
• Language auto-detection: omit the `lang` query param (or set to empty) to
  let Whisper detect the language from the first 30 seconds of audio.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from modules.asr import ASRReader, ASRResult
from modules.asr.reader import ReaderConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/asr", tags=["ASR"])

# ─── Allowed MIME types ──────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",            # MP3
    "audio/mp4",
    "audio/x-m4a",
    "audio/m4a",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/webm",
    "audio/aac",
    "audio/opus",
    "application/octet-stream",  # generic binary upload
}

# Extension whitelist as secondary guard
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".opus"}

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB (legal hearings can be long)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/process",
    response_model=ASRResult,
    status_code=status.HTTP_200_OK,
    summary="Transcribe an audio file",
    description=(
        "Upload an audio file (WAV, MP3, M4A, FLAC, OGG, WEBM, AAC). "
        "Returns a structured ASRResult with the full transcript, time-aligned segments, "
        "confidence scores, detected language, and legal intent classification."
    ),
)
async def process_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    lang: str = Query(
        "",
        description=(
            "ISO 639-1 language code hint (e.g. 'en', 'hi'). "
            "Leave empty for automatic detection."
        ),
    ),
    engine: str = Query(
        "whisper",
        description="ASR engine: 'whisper' (default) or 'faster_whisper'",
        pattern=r"^(whisper|faster_whisper)$",
    ),
    model_size: str = Query(
        "base",
        description="Whisper model size: tiny|base|small|medium|large",
        pattern=r"^(tiny|base|small|medium|large|tiny\.en|base\.en|small\.en|medium\.en)$",
    ),
) -> JSONResponse:
    # ── Validate upload size ───────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // 1_048_576} MB)",
        )

    # ── Validate content type / extension ──────────────────────────────────
    content_type = (file.content_type or "").split(";")[0].strip()
    filename = file.filename or "upload.wav"
    suffix = Path(filename).suffix.lower() or ".wav"

    if content_type not in ALLOWED_CONTENT_TYPES and suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{content_type}' / extension '{suffix}'",
        )

    # ── Write temp file ────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        config = ReaderConfig(
            language=lang if lang else None,
            engine_backend=engine,        # type: ignore[arg-type]
            model_size=model_size,
        )
        reader = ASRReader(config=config)
        result: ASRResult = reader.process(tmp_path)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ASR engine not available: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("ASR processing failed for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ASR processing failed: {exc}",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(content=result.model_dump())


@router.get(
    "/health",
    summary="Health check for ASR module",
    status_code=status.HTTP_200_OK,
)
async def health() -> JSONResponse:
    """Check dependency availability and return module status."""
    info: dict[str, object] = {"module": "asr", "status": "ok", "engines": {}}
    degraded = False

    # openai/whisper
    try:
        import whisper  # type: ignore[import]
        import torch
        info["engines"]["whisper"] = {
            "available": True,
            "version": getattr(whisper, "__version__", "unknown"),
            "torch_version": torch.__version__,
        }
    except ImportError:
        info["engines"]["whisper"] = {
            "available": False,
            "note": "pip install openai-whisper",
        }
        degraded = True

    # faster-whisper (optional)
    try:
        import faster_whisper  # type: ignore[import]
        info["engines"]["faster_whisper"] = {
            "available": True,
            "version": getattr(faster_whisper, "__version__", "unknown"),
        }
    except ImportError:
        info["engines"]["faster_whisper"] = {
            "available": False,
            "optional": True,
            "note": "pip install faster-whisper",
        }

    # soundfile (primary audio loader)
    try:
        import soundfile as sf  # type: ignore[import]
        info["engines"]["soundfile"] = {"available": True, "version": sf.__version__}
    except ImportError:
        info["engines"]["soundfile"] = {
            "available": False,
            "note": "pip install soundfile",
        }
        degraded = True

    # pydub (fallback audio loader for MP3/M4A)
    try:
        import pydub  # type: ignore[import]
        info["engines"]["pydub"] = {
            "available": True,
            "version": getattr(pydub, "version", "unknown"),
        }
    except ImportError:
        info["engines"]["pydub"] = {
            "available": False,
            "optional": True,
            "note": "pip install pydub (requires ffmpeg on PATH)",
        }

    # scipy (resampling)
    try:
        import scipy  # type: ignore[import]
        info["engines"]["scipy"] = {"available": True, "version": scipy.__version__}
    except ImportError:
        info["engines"]["scipy"] = {"available": False, "optional": True}

    if degraded:
        info["status"] = "degraded"
        return JSONResponse(content=info, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    return JSONResponse(content=info)
