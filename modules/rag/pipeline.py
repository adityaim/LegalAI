"""
modules/rag/pipeline.py
────────────────────────
RAGPipeline — the public entry point for Module 5.

Orchestrates RAGIngestor + RAGRetriever. Callers interact with this class only.

Usage::

    from modules.rag import RAGPipeline

    # Default: MockEmbedder + MockVectorStore (no heavy deps, good for tests)
    pipeline = RAGPipeline()

    # Real sentence-transformers backend
    from modules.rag.embedder import SentenceTransformerEmbedder
    from modules.rag.vector_store import ChromaVectorStore
    pipeline = RAGPipeline(
        embedder=SentenceTransformerEmbedder(),
        store=ChromaVectorStore(path="./chroma_db"),
    )

    # Ingest
    pipeline.ingest(fused_doc)

    # Query
    result = pipeline.query("Section 138 dishonoured cheque", top_k=5)
    print(result.context_for_llm)   # ready for Module 6
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunker import ChunkerConfig
from .embedder import BaseEmbedder, MockEmbedder
from .ingestor import RAGIngestor
from .retriever import RAGRetriever
from .schema import Chunk, RAGQuery, RetrievalResult
from .vector_store import BaseVectorStore, MockVectorStore


@dataclass
class PipelineConfig:
    """Top-level configuration for RAGPipeline."""
    collection: str = "legal_ai_default"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_tokens: int = 32
    default_top_k: int = 5
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    min_score: float = 0.0


class RAGPipeline:
    """
    Unified RAG pipeline: ingest FusedDocuments + retrieve for queries.

    Designed for dependency injection: pass any BaseEmbedder + BaseVectorStore.
    Defaults to MockEmbedder + MockVectorStore (zero external dependencies).
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        store: BaseVectorStore | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.embedder = embedder or MockEmbedder()
        self.store = store or MockVectorStore()

        chunker_cfg = ChunkerConfig(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            min_chunk=self.config.min_chunk_tokens,
        )
        self._ingestor = RAGIngestor(self.embedder, self.store, chunker_cfg)
        self._retriever = RAGRetriever(self.embedder, self.store)

    # ── Public API ────────────────────────────────────────────────────────

    def ingest(self, fused_doc, collection: str | None = None) -> list[Chunk]:
        """
        Chunk, embed, and store a FusedDocument.

        Parameters
        ----------
        fused_doc : FusedDocument
            Output of TextFusionProcessor.fuse().
        collection : str | None
            Override the default collection name.

        Returns
        -------
        list[Chunk]
            The chunks that were stored.
        """
        return self._ingestor.ingest(
            fused_doc,
            collection=collection or self.config.collection,
        )

    def query(
        self,
        query: str,
        top_k: int | None = None,
        collection: str | None = None,
        use_mmr: bool | None = None,
        mmr_lambda: float | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for a natural language query.

        Parameters
        ----------
        query    : str    — the legal question or context request
        top_k    : int    — number of chunks to return (default from config)
        collection : str  — which collection to search

        Returns
        -------
        RetrievalResult
            .chunks           — ranked Chunk list with scores
            .context_for_llm  — pre-formatted context string for Module 6
        """
        rag_query = RAGQuery(
            query=query,
            top_k=top_k or self.config.default_top_k,
            collection=collection or self.config.collection,
            use_mmr=use_mmr if use_mmr is not None else self.config.use_mmr,
            mmr_lambda=mmr_lambda if mmr_lambda is not None else self.config.mmr_lambda,
            min_score=min_score if min_score is not None else self.config.min_score,
        )
        return self._retriever.retrieve(rag_query)

    def collection_size(self, collection: str | None = None) -> int:
        """Return the number of chunks stored in the collection."""
        return self.store.count(collection or self.config.collection)

    def clear(self, collection: str | None = None) -> None:
        """Delete all chunks from the collection."""
        self.store.delete_collection(collection or self.config.collection)
