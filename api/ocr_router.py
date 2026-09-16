"""
api/ocr_router.py
──────────────────
FastAPI router for Module 1: OCR Document Reader.

Endpoints
─────────
POST /api/v1/ocr/process
    Upload a PDF or image file; returns OCRResult JSON.

GET  /api/v1/ocr/health
    Check that required dependencies are available.

Mount this router in your main FastAPI application::

    from api.ocr_router import router as ocr_router
    app.include_router(ocr_router)

Design notes
─────────────
• Files are written to a temporary directory and cleaned up after
  processing — no persistent storage (NFR-8 compliance).
• Reader is instantiated per-request to avoid cross-request state.
  For high-throughput, inject a shared reader via dependency injection
  with a Semaphore to limit concurrency.
• Response uses Pydantic v2 ``model_dump()`` for serialisation.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from modules.ocr import OCRDocumentReader, OCRResult, ReaderConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ocr", tags=["OCR"])

# ─── Allowed MIME types ──────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
    "image/bmp",
    # Some clients send generic binary
    "application/octet-stream",
}

# Max upload: 50 MB
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/process",
    response_model=OCRResult,
    status_code=status.HTTP_200_OK,
    summary="Process a PDF or image file",
    description=(
        "Upload a legal PDF or image file (JPEG, PNG, TIFF, WebP). "
        "Returns a structured OCRResult with extracted text, layout blocks, "
        "confidence scores, and document metadata."
    ),
)
async def process_document(
    file: UploadFile = File(..., description="PDF or image file to process"),
    lang: str = Query(
        "en",
        description="Primary language code (ISO 639-1). E.g. 'en', 'hi', 'en+hi'",
        pattern=r"^[a-z]{2}(\+[a-z]{2})*$",
    ),
    layout_engine: str = Query(
        "heuristic",
        description="Layout detection engine: 'heuristic' (default) or 'layoutparser'",
        pattern=r"^(heuristic|layoutparser)$",
    ),
    ocr_engine: str = Query(
        "paddle",
        description="OCR engine: 'paddle' (default) or 'tesseract'",
        pattern=r"^(paddle|tesseract)$",
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
            detail=(
                f"Unsupported media type '{content_type}'. "
                f"Allowed: {sorted(ALLOWED_CONTENT_TYPES - {'application/octet-stream'})}"
            ),
        )

    # ── Determine file extension ───────────────────────────────────────────
    filename = file.filename or "upload.pdf"
    suffix = Path(filename).suffix.lower() or ".pdf"

    # ── Write to temp file ────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        # ── Configure and run reader ───────────────────────────────────────
        config = ReaderConfig(
            language=lang.split("+")[0],   # primary language
            ocr_engine=ocr_engine,         # type: ignore[arg-type]
            layout_engine=layout_engine,   # type: ignore[arg-type]
        )
        reader = OCRDocumentReader(config=config)
        result: OCRResult = reader.process(tmp_path)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("OCR processing failed for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {exc}",
        ) from exc
    finally:
        # Always clean up the temp file (NFR-8)
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(content=result.model_dump())


@router.get(
    "/health",
    summary="Health check for OCR module",
    status_code=status.HTTP_200_OK,
)
async def health() -> JSONResponse:
    """
    Check dependency availability and return module status.

    Returns 200 if the module is ready; 503 if critical deps are missing.
    """
    status_info: dict[str, object] = {
        "module": "ocr",
        "status": "ok",
        "engines": {},
    }
    degraded = False

    # Check PyMuPDF
    try:
        import fitz  # noqa: F401
        status_info["engines"]["pymupdf"] = {"available": True, "version": fitz.version[0]}
    except ImportError:
        status_info["engines"]["pymupdf"] = {"available": False}
        degraded = True

    # Check PaddleOCR
    try:
        import paddleocr  # noqa: F401
        status_info["engines"]["paddleocr"] = {"available": True}
    except ImportError:
        status_info["engines"]["paddleocr"] = {"available": False}
        degraded = True  # PaddleOCR is primary — mark degraded

    # Check Tesseract (optional)
    try:
        import pytesseract
        ver = pytesseract.get_tesseract_version()
        status_info["engines"]["tesseract"] = {"available": True, "version": str(ver)}
    except Exception:
        status_info["engines"]["tesseract"] = {"available": False, "optional": True}

    # Check OpenCV
    try:
        import cv2  # noqa: F401
        status_info["engines"]["opencv"] = {"available": True, "version": cv2.__version__}
    except ImportError:
        status_info["engines"]["opencv"] = {"available": False}
        degraded = True

    if degraded:
        status_info["status"] = "degraded"
        return JSONResponse(
            content=status_info,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return JSONResponse(content=status_info)
