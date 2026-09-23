"""
modules/rag/vector_store.py
────────────────────────────
Pluggable vector store layer for Module 5 (RAG).

Implementations
───────────────
• MockVectorStore        — in-memory list; cosine similarity via pure Python. Zero deps.
• ChromaVectorStore      — wraps chromadb (persistent on disk or in-memory).
• FaissVectorStore       — wraps faiss-cpu for large-scale retrieval.

All stores implement BaseVectorStore:
    upsert(chunks, embeddings)
    query(query_embedding, top_k, collection) -> list[(chunk_id, score)]
    delete_collection(collection)
    count(collection) -> int

Reference: chromadb docs, faiss python bindings
https://docs.trychroma.com/
https://github.com/facebookresearch/faiss
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .schema import Chunk

logger = logging.getLogger(__name__)


# ─── Base interface ───────────────────────────────────────────────────────────

class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection: str = "legal_ai_default",
    ) -> None:
        """Store chunk embeddings. chunk_id is used as the unique key."""

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection: str = "legal_ai_default",
    ) -> list[tuple[str, float]]:
        """Return list of (chunk_id, cosine_similarity) sorted descending."""

    @abstractmethod
    def delete_collection(self, collection: str = "legal_ai_default") -> None:
        """Delete all chunks in a collection."""

    @abstractmethod
    def count(self, collection: str = "legal_ai_default") -> int:
        """Return number of chunks stored in collection."""

    @abstractmethod
    def get_chunk(self, chunk_id: str, collection: str = "legal_ai_default") -> Chunk | None:
        """Retrieve a single Chunk by its chunk_id."""


# ─── Cosine similarity helper ─────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# ─── Mock in-memory store ─────────────────────────────────────────────────────

@dataclass
class _Entry:
    chunk: Chunk
    embedding: list[float]


class MockVectorStore(BaseVectorStore):
    """
    In-memory vector store using pure-Python cosine similarity.

    No external dependencies. Suitable for tests and single-session use.
    Not persistent across process restarts.
    """

    def __init__(self) -> None:
        # collection_name -> list of _Entry
        self._store: dict[str, list[_Entry]] = {}

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection: str = "legal_ai_default",
    ) -> None:
        if collection not in self._store:
            self._store[collection] = []
        existing_ids = {e.chunk.chunk_id for e in self._store[collection]}
        for chunk, emb in zip(chunks, embeddings):
            if chunk.chunk_id in existing_ids:
                # Replace
                self._store[collection] = [
                    e for e in self._store[collection] if e.chunk.chunk_id != chunk.chunk_id
                ]
            self._store[collection].append(_Entry(chunk=chunk, embedding=emb))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection: str = "legal_ai_default",
    ) -> list[tuple[str, float]]:
        entries = self._store.get(collection, [])
        if not entries:
            return []
        scored = [
            (e.chunk.chunk_id, _cosine(query_embedding, e.embedding))
            for e in entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete_collection(self, collection: str = "legal_ai_default") -> None:
        self._store.pop(collection, None)

    def count(self, collection: str = "legal_ai_default") -> int:
        return len(self._store.get(collection, []))

    def get_chunk(self, chunk_id: str, collection: str = "legal_ai_default") -> Chunk | None:
        for entry in self._store.get(collection, []):
            if entry.chunk.chunk_id == chunk_id:
                return entry.chunk
        return None


# ─── ChromaDB store ───────────────────────────────────────────────────────────

class ChromaVectorStore(BaseVectorStore):
    """
    Vector store backed by ChromaDB.

    Install: pip install chromadb
    Supports: in-memory (path=None) or persistent (path="./chroma_db")
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._client = None
        self._colls: dict[str, object] = {}

    def _client_(self):
        if self._client is None:
            try:
                import chromadb  # type: ignore
                if self._path:
                    self._client = chromadb.PersistentClient(path=self._path)
                else:
                    self._client = chromadb.EphemeralClient()
                logger.info("ChromaDB client ready (path=%s)", self._path or "ephemeral")
            except ImportError as exc:
                raise ImportError("Run: pip install chromadb") from exc
        return self._client

    def _coll(self, collection: str):
        if collection not in self._colls:
            self._colls[collection] = self._client_().get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._colls[collection]

    def upsert(self, chunks, embeddings, collection="legal_ai_default"):
        coll = self._coll(collection)
        coll.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{
                "document_id": c.document_id,
                "source_modality": c.source_modality,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "has_review_flag": str(c.has_review_flag),
            } for c in chunks],
        )

    def query(self, query_embedding, top_k=5, collection="legal_ai_default"):
        coll = self._coll(collection)
        result = coll.query(query_embeddings=[query_embedding], n_results=min(top_k, self.count(collection)))
        ids = result["ids"][0]
        # Chroma returns distances (lower=more similar for cosine), convert to score
        distances = result["distances"][0]
        return [(cid, 1.0 - d) for cid, d in zip(ids, distances)]

    def delete_collection(self, collection="legal_ai_default"):
        try:
            self._client_().delete_collection(name=collection)
            self._colls.pop(collection, None)
        except Exception:
            pass

    def count(self, collection="legal_ai_default"):
        try:
            return self._coll(collection).count()
        except Exception:
            return 0

    def get_chunk(self, chunk_id: str, collection="legal_ai_default") -> Chunk | None:
        try:
            result = self._coll(collection).get(ids=[chunk_id], include=["documents", "metadatas"])
            if not result["ids"]:
                return None
            meta = result["metadatas"][0]
            return Chunk(
                chunk_id=chunk_id,
                document_id=meta["document_id"],
                source_modality=meta["source_modality"],
                text=result["documents"][0],
                chunk_index=int(meta["chunk_index"]),
                start_char=int(meta["start_char"]),
                end_char=int(meta["end_char"]),
                has_review_flag=meta["has_review_flag"] == "True",
            )
        except Exception:
            return None
