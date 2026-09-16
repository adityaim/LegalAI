"""
modules/htr/__init__.py
────────────────────────
Public API for Module 2: Handwritten Document Reader (HTR).

Usage::

    from modules.htr import HandwritingReader, HTRResult

    reader = HandwritingReader()
    result: HTRResult = reader.process("annotated_contract.png")
    print(result.full_text)
    print(result.to_fusion_text())   # for Module 4 Text Fusion
"""

from .reader import HandwritingReader, ReaderConfig
from .schema import HTRRegion, HTRResult

__all__ = [
    "HandwritingReader",
    "ReaderConfig",
    "HTRResult",
    "HTRRegion",
]

__version__ = "0.1.0"
