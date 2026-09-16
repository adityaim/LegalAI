"""
api/htr_router.py
──────────────────
FastAPI router for Module 2: Handwritten Document Reader (HTR).

Endpoints
─────────
POST /api/v1/htr/process
    Upload an image or PDF; returns HTRResult JSON with all detected
    handwritten regions, confidence scores, and bounding boxes.

GET  /api/v1/htr/health
    Check engine availability (TrOCR, Tesseract, OpenCV).

Design notes
─────────────
• Temp files are always deleted after processing (NFR-8 compliance).
• One HandwritingReader is instantiated per request; for high-throughput
  production, use a dependency-injected shared reader with a Semaphore.
• Language routing: lang=hi → Tesseract Devanagari; lang=en → TrOCR.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from modules.htr import HandwritingReader, HTRResult
from modules.htr.reader import ReaderConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/htr", tags=["HTR"])

# ─── Allowed MIME types ──────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
    "image/bmp",
    "application/octet-stream",
}

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/process",
    response_model=HTRResult,
    status_code=status.HTTP_200_OK,
    summary="Detect and recognise handwritten regions in an image or PDF",
    description=(
        "Upload a document image (JPEG, PNG, TIFF, WebP) or single-page PDF. "
        "Returns a structured HTRResult with all detected handwritten regions, "
        "their recognised text, confidence scores, bounding boxes, and handwriting type."
    ),
)
async def process_document(
    file: UploadFile = File(..., description="Image or PDF file"),
    lang: str = Query(
        "en",
        description="Primary language code (ISO 639-1). Use 'hi' for Hindi/Devanagari.",
        pattern=r"^[a-z]{2}$",
    ),
    segmenter: str = Query(
        "heuristic",
        description="Segmentation backend: 'heuristic' (default) or 'yolov8'",
        pattern=r"^(heuristic|yolov8)$",
    ),
    trocr_variant: str = Query(
        "base",
        description="TrOCR model size: 'base' (default), 'large', or 'small'",
        pattern=r"^(base|large|small)$",
    ),
    spellcheck: bool = Query(
        True,
        description="Enable SymSpell post-correction (default: true)",
    ),
) -> JSONResponse:
    # ── Validate upload size ───────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // 1_048_576} MB)",
        )

    # ── Validate content type ──────────────────────────────────────────────
    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{content_type}'",
        )

    filename = file.filename or "upload.png"
    suffix = Path(filename).suffix.lower() or ".png"

    # ── Write temp file ────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        config = ReaderConfig(
            language=lang,
            segmenter_engine=segmenter,       # type: ignore[arg-type]
            trocr_variant=trocr_variant,      # type: ignore[arg-type]
            enable_spellcheck=spellcheck,
        )
        reader = HandwritingReader(config=config)
        result: HTRResult = reader.process(tmp_path)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("HTR processing failed for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HTR processing failed: {exc}",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(content=result.model_dump())


@router.get(
    "/health",
    summary="Health check for HTR module",
    status_code=status.HTTP_200_OK,
)
async def health() -> JSONResponse:
    """Check dependency availability and return module status."""
    info: dict[str, object] = {"module": "htr", "status": "ok", "engines": {}}
    degraded = False

    # TrOCR / transformers
    try:
        import transformers  # noqa: F401
        import torch
        info["engines"]["trocr"] = {
            "available": True,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }
    except ImportError:
        info["engines"]["trocr"] = {"available": False, "note": "pip install transformers torch"}
        degraded = True

    # Tesseract Devanagari
    try:
        import pytesseract
        ver = pytesseract.get_tesseract_version()
        info["engines"]["tesseract_devanagari"] = {"available": True, "version": str(ver)}
    except Exception:
        info["engines"]["tesseract_devanagari"] = {"available": False, "optional": True}

    # OpenCV (required for segmenter)
    try:
        import cv2
        info["engines"]["opencv"] = {"available": True, "version": cv2.__version__}
    except ImportError:
        info["engines"]["opencv"] = {"available": False}
        degraded = True

    # SymSpell (optional)
    try:
        import symspellpy  # noqa: F401
        info["engines"]["symspell"] = {"available": True}
    except ImportError:
        info["engines"]["symspell"] = {"available": False, "optional": True}

    if degraded:
        info["status"] = "degraded"
        return JSONResponse(content=info, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    return JSONResponse(content=info)
