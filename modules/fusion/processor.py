"""
modules/fusion/processor.py
────────────────────────────
TextFusionProcessor — the main pipeline orchestrator for Module 4.

Pipeline overview::

    fuse(ocr, htr, asr, typed_text)
    ├── For each present modality:
    │   ├── extract raw text via to_fusion_text() / full_text / transcript
    │   ├── clean_text()             # NFKC + whitespace + legal punctuation
    │   ├── extract_review_flags()   # collect [REVIEW] spans
    │   └── build ModalityBlock
    ├── deduplicate_cross_modality() # Jaccard sentence dedup across modalities
    ├── detect_primary_language()    # majority vote from language fields
    ├── detect_intent()              # from ASR intent or heuristic fallback
    ├── wrap each block in [TAG]…[/TAG]
    ├── join into fused_text
    └── assemble FusedDocument

Tag format (mirrors the Module 4 Fusion Contract in the design doc)::

    [OCR-DOCUMENT]
    …text…
    [/OCR-DOCUMENT]

    [HTR-HANDWRITING]
    …text… [REVIEW]low-conf[/REVIEW] …
    [/HTR-HANDWRITING]

    [ASR-VOICE]
    …transcript…
    [/ASR-VOICE]

    [TYPED-INPUT]
    …typed query…
    [/TYPED-INPUT]

Typed input
───────────
A free-text string (typed query or chatbot message) can be passed directly
as `typed_text`. It is cleaned and wrapped in [TYPED-INPUT] tags.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass

from .cleaner import clean_text, deduplicate_cross_modality, extract_review_flags
from .schema import (
    FusedDocument,
    LegalIntent,
    ModalityBlock,
    ModalitySource,
    ReviewFlag,
)

logger = logging.getLogger(__name__)

# ─── Source tag map ───────────────────────────────────────────────────────────

_TAG: dict[ModalitySource, str] = {
    "OCR":   "OCR-DOCUMENT",
    "HTR":   "HTR-HANDWRITING",
    "ASR":   "ASR-VOICE",
    "TYPED": "TYPED-INPUT",
}

# Intent keyword patterns (fallback when no ASR intent is available)
_QUESTION_RE  = re.compile(r"\b(what|when|where|who|which|how|why|does|do|is|are|can)\b|\?", re.I)
_INSTRUCT_RE  = re.compile(r"\b(draft|write|prepare|send|file|submit|create|generate|compose|issue)\b", re.I)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ProcessorConfig:
    """Configuration for TextFusionProcessor."""
    enable_deduplication: bool = True
    dedup_threshold: float = 0.85      # Jaccard similarity threshold
    min_block_chars: int = 3           # blocks shorter than this are dropped
    include_review_markers: bool = True # keep [REVIEW]…[/REVIEW] in fused_text


# ─── Main processor ──────────────────────────────────────────────────────────

class TextFusionProcessor:
    """
    Fuse outputs from Modules 1–3 into a single FusedDocument.

    Usage::

        fuser = TextFusionProcessor()

        doc = fuser.fuse(
            ocr=ocr_result,     # OCRResult from Module 1
            htr=htr_result,     # HTRResult from Module 2
            asr=asr_result,     # ASRResult from Module 3
            typed_text="What is Section 138?",
        )

        print(doc.fused_text)
        print(doc.for_rag())    # tag-stripped for vector embedding
        print(doc.for_llm())    # with metadata header for LLM context
    """

    def __init__(self, config: ProcessorConfig | None = None) -> None:
        self.config = config or ProcessorConfig()

    # ── Public API ────────────────────────────────────────────────────────

    def fuse(
        self,
        ocr=None,        # OCRResult | None
        htr=None,        # HTRResult | None
        asr=None,        # ASRResult | None
        typed_text: str | None = None,
    ) -> FusedDocument:
        """
        Merge all provided modality results into a FusedDocument.

        At least one of ocr / htr / asr / typed_text must be non-None.

        Parameters
        ----------
        ocr : OCRResult | None
            Output from Module 1 (OCR Document Reader).
        htr : HTRResult | None
            Output from Module 2 (HTR Handwriting Reader).
        asr : ASRResult | None
            Output from Module 3 (ASR Voice-to-Text).
        typed_text : str | None
            Free-text typed query or chatbot message.

        Returns
        -------
        FusedDocument
        """
        if not any([ocr, htr, asr, typed_text]):
            raise ValueError("At least one input modality must be provided.")

        start = time.perf_counter()
        doc_id = str(uuid.uuid4())
        source_ids: list[str] = []
        all_flags: list[ReviewFlag] = []

        # ── Extract raw text + metadata per modality ───────────────────────
        raw_blocks: list[tuple[ModalitySource, str, str]] = []
        # → (source, raw_text, language)

        if ocr is not None:
            source_ids.append(getattr(ocr, "document_id", ""))
            raw_text = ocr.full_text or ""
            lang = getattr(ocr, "language", "en")
            raw_blocks.append(("OCR", raw_text, lang))

        if htr is not None:
            source_ids.append(getattr(htr, "document_id", ""))
            raw_text = htr.to_fusion_text() if hasattr(htr, "to_fusion_text") else (htr.full_text or "")
            lang = getattr(htr, "language", "en")
            raw_blocks.append(("HTR", raw_text, lang))

        if asr is not None:
            source_ids.append(getattr(asr, "document_id", ""))
            raw_text = asr.to_fusion_text() if hasattr(asr, "to_fusion_text") else (asr.transcript or "")
            lang = getattr(asr, "language", "en")
            raw_blocks.append(("ASR", raw_text, lang))

        if typed_text:
            raw_blocks.append(("TYPED", typed_text, "en"))

        # ── Clean each block ───────────────────────────────────────────────
        cleaned: list[tuple[ModalitySource, str, str]] = []
        for source, text, lang in raw_blocks:
            ct = clean_text(text)
            if len(ct) >= self.config.min_block_chars:
                cleaned.append((source, ct, lang))
            else:
                logger.debug("Dropping empty/tiny %s block (%d chars)", source, len(ct))

        # ── Cross-modality deduplication ───────────────────────────────────
        if self.config.enable_deduplication and len(cleaned) > 1:
            texts_only = [(s, t) for s, t, _ in cleaned]
            deduped = deduplicate_cross_modality(texts_only)
            cleaned = [(s, d, l) for (s, _, l), d in zip(cleaned, deduped)]

        # ── Build ModalityBlocks + extract review flags ────────────────────
        modality_blocks: list[ModalityBlock] = []
        for source, text, lang in cleaned:
            flags = extract_review_flags(text, source)
            all_flags.extend(flags)
            modality_blocks.append(ModalityBlock(
                source=source,
                text=text,
                language=lang,
                has_review_flags=bool(flags),
            ))

        # ── Assemble fused_text ────────────────────────────────────────────
        parts: list[str] = []
        for block in modality_blocks:
            tag = _TAG[block.source]
            parts.append(f"[{tag}]\n{block.text}\n[/{tag}]")

        fused_text = "\n\n".join(parts)

        # ── Language detection ─────────────────────────────────────────────
        primary_lang = self._detect_language(cleaned, asr)

        # ── Intent detection ───────────────────────────────────────────────
        intent = self._detect_intent(asr, fused_text)

        elapsed = time.perf_counter() - start
        logger.info(
            "Fusion complete: %d modalities, %d words, intent=%s, lang=%s, %.3fs",
            len(modality_blocks),
            len(fused_text.split()),
            intent,
            primary_lang,
            elapsed,
        )

        return FusedDocument(
            document_id=doc_id,
            source_ids=[s for s in source_ids if s],
            modalities=modality_blocks,
            fused_text=fused_text,
            primary_language=primary_lang,
            detected_intent=intent,
            review_flags=all_flags,
            processing_time_s=round(elapsed, 4),
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _detect_language(
        self,
        cleaned: list[tuple[ModalitySource, str, str]],
        asr=None,
    ) -> str:
        """
        Determine primary language via majority vote across all modality langs.

        ASR language detection is weighted 2× (Whisper is reliable for lang ID).
        """
        lang_votes: list[str] = []
        lang_map = {s: l for s, _, l in cleaned}

        for source, lang in lang_map.items():
            weight = 2 if source == "ASR" else 1
            lang_votes.extend([lang] * weight)

        if not lang_votes:
            return "en"
        counter = Counter(lang_votes)
        return counter.most_common(1)[0][0]

    def _detect_intent(self, asr, fused_text: str) -> LegalIntent:
        """
        Inherit intent from ASR if available; otherwise run heuristic on fused text.
        """
        if asr is not None:
            asr_intent = getattr(asr, "intent", "unknown")
            if asr_intent != "unknown":
                return asr_intent  # type: ignore[return-value]

        # Heuristic fallback on typed input or full fused text
        if _QUESTION_RE.search(fused_text):
            return "question"
        if _INSTRUCT_RE.search(fused_text):
            return "instruction"
        if fused_text.strip():
            return "narration"
        return "unknown"
