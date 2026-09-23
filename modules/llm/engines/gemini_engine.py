"""
modules/llm/engines/gemini_engine.py
──────────────────────────────────────
GeminiEngine: wraps Google Gemini via the google-genai SDK.

Install: pip install google-genai
Requires: GEMINI_API_KEY environment variable.

Model: gemini-2.0-flash (default) — fast, cost-effective for legal QA.
       Can be overridden via constructor.

Confidence heuristic:
  - If the answer contains [UNCERTAIN] markers -> confidence = 60.0
  - Otherwise -> confidence = 85.0
  (Full probability-based confidence would require logprob access,
   which is not available in all Gemini API tiers.)

Reference: Google GenAI Python SDK
https://github.com/googleapis/python-genai
"""

from __future__ import annotations

import logging
import os

from .base import BaseLLMEngine, LLMResponse

logger = logging.getLogger(__name__)

_UNCERTAIN_MARKER = "[UNCERTAIN]"


class GeminiEngine(BaseLLMEngine):
    """
    LLM engine backed by Google Gemini API.

    Usage::

        from modules.llm.engines.gemini_engine import GeminiEngine
        engine = GeminiEngine()          # uses GEMINI_API_KEY env var
        engine = GeminiEngine(model="gemini-1.5-pro")
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client = None  # lazy-loaded

    @property
    def engine_name(self) -> str:
        return self._model

    def _load(self) -> None:
        if self._client is not None:
            return
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            ) from exc
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Export it before using GeminiEngine."
            )
        self._client = genai.Client(api_key=self._api_key)
        logger.info("GeminiEngine ready: model=%s", self._model)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self._load()

        from google.genai import types  # type: ignore

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=0.2,       # low temperature for factual legal answers
                top_p=0.8,
            ),
        )

        text = response.text or ""
        confidence = 60.0 if _UNCERTAIN_MARKER in text else 85.0

        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = getattr(response.usage_metadata, "total_token_count", 0) or 0

        logger.info(
            "GeminiEngine generated %d chars (%d tokens), confidence=%.1f",
            len(text), tokens_used, confidence,
        )

        return LLMResponse(
            text=text,
            confidence=confidence,
            engine_name=self.engine_name,
            tokens_used=tokens_used,
        )

    def warmup(self) -> None:
        """Pre-load the client (no model weights to download for API calls)."""
        self._load()
