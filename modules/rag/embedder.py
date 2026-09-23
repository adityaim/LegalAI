"""
modules/rag/embedder.py
────────────────────────
Pluggable embedding layer for Module 5 (RAG).

Implementations
───────────────
• MockEmbedder         — deterministic hash-based vectors; zero dependencies; used in all tests.
• SentenceTransformerEmbedder — wraps `sentence-transformers` (all-MiniLM-L6-v2 default).
• GeminiEmbedder       — wraps Google's text-embedding-004 via google-genai SDK.

All embedders implement the same BaseEmbedder interface:
    embed(texts: list[str]) -> list[list[float]]

The embedding dimension is fixed per model:
  MockEmbedder:             128  (configurable)
  all-MiniLM-L6-v2:         384
  text-embedding-004:      768

Reference: sentence-transformers repo
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ─── Base interface ───────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Abstract embedding interface. Implement embed() to plug in any model."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Parameters
        ----------
        texts : list[str]
            Non-empty list of text strings to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text.
            All vectors must have the same dimension.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimensionality."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""


# ─── Mock embedder (test/CI) ──────────────────────────────────────────────────

class MockEmbedder(BaseEmbedder):
    """
    Deterministic hash-based embedder. Zero dependencies, zero latency.

    Produces normalised float32 vectors from SHA-256 of the text.
    Semantically similar texts will NOT cluster (by design for a mock),
    but the vectors are deterministic and correct-dimensioned, making
    them suitable for all unit and integration tests.
    """

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"mock-{self._dim}d"

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Repeat digest bytes to fill dim floats
            raw = []
            while len(raw) < self._dim:
                raw.extend(digest)
            floats = [b / 255.0 - 0.5 for b in raw[: self._dim]]
            # L2-normalise
            norm = math.sqrt(sum(f * f for f in floats)) or 1.0
            results.append([f / norm for f in floats])
        return results


# ─── sentence-transformers embedder ──────────────────────────────────────────

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder backed by `sentence-transformers`.

    Install: pip install sentence-transformers
    Default model: all-MiniLM-L6-v2 (384-dim, ~80 MB, Apache-2.0)
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model = None   # lazy-loaded

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(self._model_name, device=self._device)
                logger.info("Loaded sentence-transformer: %s on %s", self._model_name, self._device)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()


# ─── Gemini embedder ─────────────────────────────────────────────────────────

class GeminiEmbedder(BaseEmbedder):
    """
    Embedder backed by Google's text-embedding-004 via the google-genai SDK.

    Install: pip install google-genai
    Requires: GEMINI_API_KEY environment variable.
    Dimension: 768 (configurable via output_dimensionality).
    """

    DEFAULT_MODEL = "models/text-embedding-004"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = 768,
    ) -> None:
        self._model_name = model_name
        self._task_type = task_type
        self._dim = output_dimensionality
        self._client = None   # lazy-loaded

    def _load(self) -> None:
        if self._client is None:
            try:
                import google.generativeai as genai  # type: ignore
                import os
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable not set")
                genai.configure(api_key=api_key)
                self._client = genai
                logger.info("Gemini embedder ready: %s", self._model_name)
            except ImportError as exc:
                raise ImportError("Run: pip install google-generativeai") from exc

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        results = []
        for text in texts:
            resp = self._client.embed_content(
                model=self._model_name,
                content=text,
                task_type=self._task_type,
                output_dimensionality=self._dim,
            )
            results.append(resp["embedding"])
        return results
