"""
modules/fusion/__init__.py
───────────────────────────
Public API surface for Module 4: Text Fusion & Pre-processing.

Usage::

    from modules.fusion import TextFusionProcessor, FusedDocument

    fuser = TextFusionProcessor()

    # Fuse from individual modality results
    doc = fuser.fuse(ocr=ocr_result, htr=htr_result, asr=asr_result)

    # Or from a typed query string only
    doc = fuser.fuse(typed_text="What is Section 138 NI Act?")

    print(doc.fused_text)          # tagged, clean context block
    print(doc.for_rag())           # plain text ready for vector embedding
    print(doc.primary_language)    # "en" | "hi" | ...
    print(doc.detected_intent)     # "question" | "instruction" | ...
"""

from .processor import TextFusionProcessor, ProcessorConfig
from .schema import FusedDocument, ModalityBlock

__all__ = [
    "TextFusionProcessor",
    "ProcessorConfig",
    "FusedDocument",
    "ModalityBlock",
]
