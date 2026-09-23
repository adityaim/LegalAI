"""
modules/rag/__init__.py
────────────────────────
Public API surface for Module 5: RAG Ingestion & Retrieval.

Usage::

    from modules.rag import RAGPipeline

    pipeline = RAGPipeline()           # uses mock store by default in tests

    # Ingest a FusedDocument from Module 4
    pipeline.ingest(fused_doc)

    # Query
    result = pipeline.query("Section 138 NI Act dishonoured cheque", top_k=5)
    for chunk in result.chunks:
        print(chunk.score, chunk.text)
"""

from .pipeline import RAGPipeline, PipelineConfig
from .schema import Chunk, RetrievalResult, RAGQuery

__all__ = [
    "RAGPipeline",
    "PipelineConfig",
    "Chunk",
    "RetrievalResult",
    "RAGQuery",
]
