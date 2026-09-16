"""modules/ocr/engines/__init__.py"""
from .base import OCREngine, WordResult
from .paddle_engine import PaddleOCREngine
from .tesseract_engine import TesseractEngine

__all__ = ["OCREngine", "WordResult", "PaddleOCREngine", "TesseractEngine"]
