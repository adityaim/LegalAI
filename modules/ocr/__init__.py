"""
modules/ocr/__init__.py
────────────────────────
Public API surface for the OCR module.

Consumers of this module should import from here rather than from
sub-modules directly, to maintain a stable interface.

Example::

    from modules.ocr import OCRDocumentReader, OCRResult

    reader = OCRDocumentReader()
    result: OCRResult = reader.process("path/to/contract.pdf")
"""

from .reader import OCRDocumentReader, ReaderConfig
from .schema import BoundingBox, DocumentMetadata, OCRBlock, OCRResult

__all__ = [
    # Main pipeline class
    "OCRDocumentReader",
    "ReaderConfig",
    # Schema types (for type hints in downstream modules)
    "OCRResult",
    "OCRBlock",
    "BoundingBox",
    "DocumentMetadata",
]

__version__ = "0.1.0"
