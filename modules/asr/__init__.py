"""
modules/asr/__init__.py
────────────────────────
Public API surface for Module 3: ASR Voice-to-Text.

Usage::

    from modules.asr import ASRReader, ASRResult
    reader = ASRReader()
    result = reader.process("hearing.mp3")
    print(result.transcript)
    print(result.language)          # "en" | "hi" | ...
    print(result.to_fusion_text())  # for Module 4
"""

from .reader import ASRReader, ReaderConfig
from .schema import ASRResult, ASRSegment

__all__ = [
    "ASRReader",
    "ReaderConfig",
    "ASRResult",
    "ASRSegment",
]
