"""
api/llm_router.py
──────────────────
FastAPI router for Module 6: LLM Legal Reasoning.

Endpoints
─────────
POST /api/v1/llm/reason
    body: LegalQuery JSON
    Returns LegalAnswer JSON.

GET  /api/v1/llm/health
    Returns engine name, config, and readiness.

Design note: module-level singleton LegalReasoner with MockLLMEngine by default.
In production, replace engine via LEGAL_AI_LLM_ENGINE env var ("gemini").
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from modules.llm import LegalReasoner
from modules.llm.schema import LegalAnswer, LegalQuery

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])

_reasoner: LegalReasoner | None = None


def _get_reasoner() -> LegalReasoner:
    global _reasoner
    if _reasoner is None:
        engine_name = os.environ.get("LEGAL_AI_LLM_ENGINE", "mock").lower()
        if engine_name == "gemini":
            from modules.llm.engines.gemini_engine import GeminiEngine
            engine = GeminiEngine()
        else:
            from modules.llm.engines.mock_engine import MockLLMEngine
            engine = MockLLMEngine()
        _reasoner = LegalReasoner(engine=engine)
        logger.info("LegalReasoner initialised with engine=%s", engine.engine_name)
    return _reasoner


@router.post(
    "/reason",
    response_model=LegalAnswer,
    summary="Run LLM legal reasoning on a query with fused context",
)
async def reason(body: LegalQuery) -> JSONResponse:
    try:
        answer = _get_reasoner().reason(body)
    except Exception as exc:
        logger.exception("Reasoning failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(content=answer.model_dump())


@router.get("/health", summary="Health check for LLM module")
async def health() -> JSONResponse:
    reasoner = _get_reasoner()
    return JSONResponse(content={
        "module": "llm",
        "status": "ok",
        "engine": reasoner.engine.engine_name,
        "confidence_threshold": reasoner.config.confidence_threshold,
    })
