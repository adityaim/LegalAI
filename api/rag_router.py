"""
api/rag_router.py
──────────────────
FastAPI router for Module 5: RAG Ingestion & Retrieval.

Endpoints
─────────
POST /api/v1/rag/ingest
    Accept a serialised FusedDocument JSON dict.
    Chunk → embed → store. Returns list of stored chunk IDs.

POST /api/v1/rag/query
    Accept RAGQuery JSON.
    Returns RetrievalResult with ranked chunks + context_for_llm.

GET  /api/v1/rag/health
    Returns collection size and embedder/store info.

Design notes
─────────────
• The router maintains a module-level RAGPipeline singleton (mock by default).
  In production, replace with ChromaDB + SentenceTransformers via env vars.
• Collection name defaults to "legal_ai_default"; callers may override per-request.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from modules.fusion.schema import FusedDocument
from modules.rag import RAGPipeline
from modules.rag.schema import RAGQuery, RetrievalResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

# Module-level pipeline singleton (mock by default; override in production)
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        logger.info("RAGPipeline initialised with MockEmbedder + MockVectorStore")
    return _pipeline


# ─── Ingest ───────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a FusedDocument into the vector store",
)
async def ingest(body: dict) -> JSONResponse:
    try:
        fused_doc = FusedDocument.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid FusedDocument: {exc}") from exc

    try:
        chunks = _get_pipeline().ingest(fused_doc)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        status_code=201,
        content={
            "document_id": fused_doc.document_id,
            "chunks_stored": len(chunks),
            "chunk_ids": [c.chunk_id for c in chunks],
        },
    )


# ─── Query ────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=RetrievalResult,
    summary="Query the vector store for relevant legal context chunks",
)
async def query(body: RAGQuery) -> JSONResponse:
    try:
        result = _get_pipeline().query(
            query=body.query,
            top_k=body.top_k,
            collection=body.collection,
            use_mmr=body.use_mmr,
            mmr_lambda=body.mmr_lambda,
            min_score=body.min_score,
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(content=result.model_dump())


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", summary="Health check for RAG module")
async def health() -> JSONResponse:
    pipeline = _get_pipeline()
    return JSONResponse(content={
        "module": "rag",
        "status": "ok",
        "embedder": pipeline.embedder.model_name,
        "vector_store": type(pipeline.store).__name__,
        "collection_size": pipeline.collection_size(),
    })
