"""
modules/llm/engines/base.py
────────────────────────────
Abstract LLM engine interface for Module 6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """
    Raw response from an LLM engine.

    text        — the generated text.
    confidence  — self-reported or heuristic confidence (0–100).
    engine_name — identifier string for the engine used.
    tokens_used — approximate token count (0 if unavailable).
    """
    text: str
    confidence: float
    engine_name: str
    tokens_used: int = 0


class BaseLLMEngine(ABC):
    """
    Abstract base for all LLM engines.

    Implement generate() to plug in any LLM (Gemini, OpenAI, Ollama, …).
    The default warmup() is a no-op; override for model pre-loading.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response given system + user prompts."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Short identifier string, e.g. 'mock', 'gemini-2.0-flash'."""

    def warmup(self) -> None:
        """Optional: pre-load model weights or warm up connection pool."""
