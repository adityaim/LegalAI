"""
api/main.py
─────────────
FastAPI application entry point for the Legal AI document processing service.

Modules mounted:
  • Module 1 (OCR)  → /api/v1/ocr/...
  • Module 2 (HTR)  → /api/v1/htr/...

Run locally::

    uvicorn api.main:app --reload --port 8001

Or via Docker::

    docker build -f Dockerfile.ocr -t legal-ai .
    docker run -p 8001:8001 legal-ai

API docs::

    http://localhost:8001/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ocr_router import router as ocr_router
from api.htr_router import router as htr_router
from api.asr_router import router as asr_router
from api.fusion_router import router as fusion_router
from api.rag_router import router as rag_router
from api.llm_router import router as llm_router
from api.response_router import router as response_router

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

_log = logging.getLogger(__name__)


# ─── Application ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Startup / shutdown lifecycle handler."""
    _log.info("Legal AI Document Processing service starting…")
    _log.info("  Module 1 (OCR)      : /api/v1/ocr/")
    _log.info("  Module 2 (HTR)      : /api/v1/htr/")
    _log.info("  Module 3 (ASR)      : /api/v1/asr/")
    _log.info("  Module 4 (Fusion)   : /api/v1/fusion/")
    _log.info("  Module 5 (RAG)      : /api/v1/rag/")
    _log.info("  Module 6 (LLM)      : /api/v1/llm/")
    _log.info("  Module 7 (Response) : /api/v1/response/")
    yield
    _log.info("Legal AI Document Processing service stopping…")


app = FastAPI(
    title="Legal AI — Document Processing Service",
    description=(
        "Multimodal Legal Assistant System — Full Pipeline.\n\n"
        "**Module 1 (OCR):** Converts scanned legal PDFs and images to structured text.\n\n"
        "**Module 2 (HTR):** Detects and recognises handwritten regions in legal documents.\n\n"
        "**Module 3 (ASR):** Transcribes audio/voice recordings of legal proceedings.\n\n"
        "**Module 4 (Fusion):** Merges OCR/HTR/ASR outputs into a unified legal context document.\n\n"
        "**Module 5 (RAG):** Chunks, embeds, stores, and retrieves legal context for LLM reasoning.\n\n"
        "**Module 6 (LLM):** Runs grounded legal reasoning via Gemini/Mock LLM with citations.\n\n"
        "**Module 7 (Response):** Formats LegalAnswer into Markdown / Plain / HTML for end users."
    ),
    version="0.7.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins in development; restrict in production via env vars
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(ocr_router)
app.include_router(htr_router)
app.include_router(asr_router)
app.include_router(fusion_router)
app.include_router(rag_router)
app.include_router(llm_router)
app.include_router(response_router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "legal-ai-document-processing",
        "version": "0.7.0",
        "docs": "/docs",
        "modules": {
            "ocr":      "/api/v1/ocr/health",
            "htr":      "/api/v1/htr/health",
            "asr":      "/api/v1/asr/health",
            "fusion":   "/api/v1/fusion/health",
            "rag":      "/api/v1/rag/health",
            "llm":      "/api/v1/llm/health",
            "response": "/api/v1/response/health",
        },
    }
