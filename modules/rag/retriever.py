"""
modules/rag/retriever.py
─────────────────────────
Retriever for Module 5 (RAG). Sits between the vector store and the pipeline.

Responsibilities
────────────────
1. Embed the query text.
2. Run approximate nearest-neighbour search in the vector store.
3. Hydrate chunk_ids back to Chunk objects (with scores attached).
4. Apply MMR (Maximal Marginal Relevance) reranking if enabled.
5. Apply min_score threshold filtering.

MMR Algorithm
─────────────
MMR balances relevance (score vs query) and diversity (dissimilarity from
already-selected chunks). At each step it picks the chunk maximising:

    MMR(d) = lambda * sim(d, query) - (1 - lambda) * max(sim(d, selected))

where lambda=1 gives pure relevance, lambda=0 gives pure diversity.

Reference: Carbonell & Goldstein 1998, "The Use of MMR, Diversity-Based
Reranking for Reordering Documents and Producing Summaries", SIGIR.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from .embedder import BaseEmbedder
from .schema import Chunk, RAGQuery, RetrievalResult
from .vector_store import BaseVectorStore, _cosine

logger = logging.getLogger(__name__)


def _mmr_rerank(
    candidates: list[Chunk],
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    top_k: int,
    lambda_: float = 0.5,
) -> list[Chunk]:
    """
    Maximal Marginal Relevance reranking.

    Returns top_k chunks that balance relevance and diversity.
    """
    if len(candidates) <= top_k:
        return candidates

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    while len(selected_indices) < top_k and remaining:
        mmr_scores: list[tuple[int, float]] = []
        for idx in remaining:
            rel = candidates[idx].score or 0.0
            if selected_indices:
                max_sim = max(
                    _cosine(candidate_embeddings[idx], candidate_embeddings[s])
                    for s in selected_indices
                )
            else:
                max_sim = 0.0
            mmr = lambda_ * rel - (1 - lambda_) * max_sim
            mmr_scores.append((idx, mmr))

        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


class RAGRetriever:
    """
    Retrieves and reranks chunks from the vector store for a given query.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
    ) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, rag_query: RAGQuery) -> RetrievalResult:
        """
        End-to-end retrieval for a RAGQuery.

        1. Embed query text.
        2. Query vector store for top_k * 3 candidates (over-fetch for MMR).
        3. Hydrate chunk objects with scores.
        4. Apply MMR if enabled.
        5. Filter by min_score.
        6. Return RetrievalResult.
        """
        t0 = time.perf_counter()

        # 1. Embed query
        query_emb = self.embedder.embed([rag_query.query])[0]

        # 2. Over-fetch for MMR diversity pool
        fetch_k = rag_query.top_k * 3
        raw_results = self.store.query(
            query_embedding=query_emb,
            top_k=fetch_k,
            collection=rag_query.collection,
        )

        total_searched = self.store.count(rag_query.collection)

        # 3. Hydrate chunks
        chunks: list[Chunk] = []
        chunk_embeddings: list[list[float]] = []
        for chunk_id, score in raw_results:
            chunk = self.store.get_chunk(chunk_id, rag_query.collection)
            if chunk is None:
                continue
            # Clamp: negative cosine similarity means no relevance
            clamped_score = round(max(0.0, float(score)), 4)
            chunk = chunk.model_copy(update={"score": clamped_score})
            chunks.append(chunk)
            # Re-embed for MMR (or reuse query_emb cache in real impl)
            chunk_embeddings.append(self.embedder.embed([chunk.text])[0])

        # 4. MMR reranking
        if rag_query.use_mmr and len(chunks) > rag_query.top_k:
            chunks = _mmr_rerank(
                chunks, query_emb, chunk_embeddings,
                top_k=rag_query.top_k,
                lambda_=rag_query.mmr_lambda,
            )
        else:
            chunks = chunks[: rag_query.top_k]

        # 5. Min-score filter
        if rag_query.min_score > 0.0:
            chunks = [c for c in chunks if (c.score or 0.0) >= rag_query.min_score]

        elapsed = time.perf_counter() - t0
        logger.info(
            "Retrieved %d chunks for query=%r in %.3fs (pool=%d, total=%d)",
            len(chunks), rag_query.query[:60], elapsed, len(raw_results), total_searched,
        )

        return RetrievalResult(
            chunks=chunks,
            query=rag_query.query,
            total_chunks_searched=total_searched,
            retrieval_time_s=round(elapsed, 4),
        )
