"""
api/fusion_router.py
─────────────────────
FastAPI router for Module 4: Text Fusion & Pre-processing.

Endpoints
─────────
POST /api/v1/fusion/fuse
    Accept JSON with any combination of:
      • ocr_result   — serialised OCRResult (from Module 1)
      • htr_result   — serialised HTRResult (from Module 2)
      • asr_result   — serialised ASRResult (from Module 3)
      • typed_text   — free-text string (typed query)
    Returns FusedDocument JSON.

GET  /api/v1/fusion/health
    Check module availability.

Design notes
─────────────
• Accepts modality results as raw JSON dicts (model_validate) so the caller
  does not need to import the schema classes directly.
• At least one of the four inputs must be present.
• typed_text can be used standalone (e.g. chatbot message only).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

from modules.fusion import FusedDocument, TextFusionProcessor
from modules.fusion.processor import ProcessorConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/fusion", tags=["Fusion"])


# ─── Request model ────────────────────────────────────────────────────────────

class FuseRequest(BaseModel):
    """
    Request body for POST /api/v1/fusion/fuse.

    Supply any combination of modality result dicts + optional typed text.
    At least one field must be non-null.
    """
    ocr_result:  dict | None = None
    htr_result:  dict | None = None
    asr_result:  dict | None = None
    typed_text:  str  | None = None

    enable_deduplication: bool = True
    min_block_chars: int = 3

    @model_validator(mode="after")
    def at_least_one_input(self) -> "FuseRequest":
        if not any([self.ocr_result, self.htr_result, self.asr_result, self.typed_text]):
            raise ValueError(
                "At least one of ocr_result, htr_result, asr_result, or typed_text must be provided."
            )
        return self


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/fuse",
    response_model=FusedDocument,
    status_code=status.HTTP_200_OK,
    summary="Fuse modality results into a unified legal context document",
    description=(
        "Accepts any combination of OCRResult, HTRResult, ASRResult (as JSON dicts) "
        "and/or a typed text string. Returns a FusedDocument with a tagged composite "
        "text, detected language, legal intent, and review flags."
    ),
)
async def fuse_modalities(body: FuseRequest) -> JSONResponse:
    from modules.ocr.schema import OCRResult
    from modules.htr.schema import HTRResult
    from modules.asr.schema import ASRResult

    # Deserialise each provided result
    ocr = OCRResult.model_validate(body.ocr_result) if body.ocr_result else None
    htr = HTRResult.model_validate(body.htr_result) if body.htr_result else None
    asr = ASRResult.model_validate(body.asr_result) if body.asr_result else None

    config = ProcessorConfig(
        enable_deduplication=body.enable_deduplication,
        min_block_chars=body.min_block_chars,
    )

    try:
        fuser = TextFusionProcessor(config=config)
        doc: FusedDocument = fuser.fuse(
            ocr=ocr, htr=htr, asr=asr, typed_text=body.typed_text
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Fusion failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fusion failed: {exc}",
        ) from exc

    return JSONResponse(content=doc.model_dump())


@router.get(
    "/health",
    summary="Health check for Fusion module",
    status_code=status.HTTP_200_OK,
)
async def health() -> JSONResponse:
    info: dict[str, object] = {"module": "fusion", "status": "ok", "dependencies": {}}

    for pkg in ("unicodedata", "re", "collections"):
        info["dependencies"][pkg] = {"available": True, "builtin": True}

    return JSONResponse(content=info)
