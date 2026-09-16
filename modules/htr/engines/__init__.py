"""modules/htr/engines/__init__.py"""
from .base import HTREngine, RegionRecognition
from .trocr_engine import TrOCREngine
from .tesseract_devanagari import TesseractDevanagariEngine

__all__ = [
    "HTREngine",
    "RegionRecognition",
    "TrOCREngine",
    "TesseractDevanagariEngine",
]
