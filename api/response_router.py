"""
api/response_router.py
───────────────────────
FastAPI router for Module 7: Response Generation & Formatting.

Endpoints
─────────
POST /api/v1/response/format
    body: { "legal_answer": <LegalAnswer dict>, "format": "markdown"|"plain"|"html"|"json" }
    Returns FormattedResponse JSON.

GET  /api/v1/response/health
    Returns formatter config.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from modules.llm.schema import LegalAnswer
from modules.response import ResponseFormatter
from modules.response.schema import FormattedResponse, ResponseFormat

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/response", tags=["Response"])

_formatter: ResponseFormatter | None = None


def _get_formatter() -> ResponseFormatter:
    global _formatter
    if _formatter is None:
        _formatter = ResponseFormatter()
        logger.info("ResponseFormatter initialised")
    return _formatter


class _FormatRequest(LegalAnswer):
    """Extends LegalAnswer to add a format field for the request body."""
    pass


@router.post(
    "/format",
    response_model=FormattedResponse,
    summary="Format a LegalAnswer into a user-facing response",
)
async def format_answer(body: dict) -> JSONResponse:
    fmt_str = body.pop("format", "markdown")
    try:
        fmt = ResponseFormat(fmt_str.lower())
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown format: {fmt_str!r}")

    try:
        legal_answer = LegalAnswer.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid LegalAnswer: {exc}") from exc

    try:
        response = _get_formatter().format(legal_answer, fmt=fmt)
    except Exception as exc:
        logger.exception("Formatting failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(content=response.model_dump())


@router.get("/health", summary="Health check for Response module")
async def health() -> JSONResponse:
    fmt = _get_formatter()
    return JSONResponse(content={
        "module": "response",
        "status": "ok",
        "default_format": fmt.config.default_format.value,
        "max_citations": fmt.config.max_citations,
        "strip_internal_markers": fmt.config.strip_internal_markers,
    })
