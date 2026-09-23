"""
modules/rag/ingestor.py
────────────────────────
RAGIngestor — converts a FusedDocument into stored vector embeddings.

Pipeline:
    FusedDocument.for_rag()
        -> SentenceAwareChunker.chunk()   # split into Chunk objects
        -> BaseEmbedder.embed()           # generate float vectors
        -> BaseVectorStore.upsert()       # persist
"""

from __future__ import annotations

import logging
import time

from .chunker import SentenceAwareChunker, ChunkerConfig
from .embedder import BaseEmbedder
from .schema import Chunk
from .vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class RAGIngestor:
    """
    Ingests a FusedDocument into the vector store.

    Usage::

        ingestor = RAGIngestor(embedder=embedder, store=store)
        chunks_stored = ingestor.ingest(fused_doc, collection="legal_ai_default")
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        chunker_config: ChunkerConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = SentenceAwareChunker(chunker_config or ChunkerConfig())

    def ingest(
        self,
        fused_doc,          # FusedDocument (avoid circular import; duck-typed)
        collection: str = "legal_ai_default",
    ) -> list[Chunk]:
        """
        Chunk, embed, and store a FusedDocument.

        Parameters
        ----------
        fused_doc : FusedDocument
            Output of TextFusionProcessor.fuse().
        collection : str
            Vector store collection name.

        Returns
        -------
        list[Chunk]
            The chunks that were embedded and stored.
        """
        t0 = time.perf_counter()

        rag_text    = fused_doc.for_rag()
        fused_text  = fused_doc.fused_text
        document_id = fused_doc.document_id

        # 1. Chunk
        chunks = self.chunker.chunk(
            rag_text=rag_text,
            fused_text=fused_text,
            document_id=document_id,
        )

        if not chunks:
            logger.warning("No chunks produced for document_id=%s", document_id)
            return []

        # 2. Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        # 3. Store
        self.store.upsert(chunks=chunks, embeddings=embeddings, collection=collection)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Ingested %d chunks from document_id=%s into collection=%r in %.3fs",
            len(chunks), document_id, collection, elapsed,
        )

        return chunks
