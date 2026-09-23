"""
tests/fusion/test_schema.py
────────────────────────────
Unit tests for the Fusion output schema (Module 4).

Tests cover:
  • ModalityBlock construction, word_count auto-derivation
  • FusedDocument construction, word_count auto-derivation
  • for_rag()  — tag stripping, REVIEW marker preservation
  • for_llm()  — metadata header, fused_text present
  • has_content() — empty vs non-empty
  • modality_text() — lookup by source
  • ReviewFlag construction
  • Serialisation round-trip (model_dump -> model_validate)
  • Invalid intents / sources rejected by Pydantic
"""

import pytest
from pydantic import ValidationError

from modules.fusion.schema import (
    FusedDocument,
    ModalityBlock,
    ReviewFlag,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_block(source="OCR", text="The accused appeared before the court.", lang="en") -> ModalityBlock:
    return ModalityBlock(source=source, text=text, language=lang)  # type: ignore[arg-type]


def make_doc(
    fused_text: str = "[OCR-DOCUMENT]\nTest text here.\n[/OCR-DOCUMENT]",
    intent: str = "narration",
    modalities: list | None = None,
) -> FusedDocument:
    blocks = modalities or [make_block()]
    return FusedDocument(
        modalities=blocks,
        fused_text=fused_text,
        primary_language="en",
        detected_intent=intent,           # type: ignore[arg-type]
        processing_time_s=0.01,
    )


# ─── ModalityBlock ────────────────────────────────────────────────────────────

class TestModalityBlock:
    def test_valid_ocr(self):
        b = make_block("OCR")
        assert b.source == "OCR"
        assert b.language == "en"

    def test_word_count_auto_derived(self):
        b = make_block(text="One two three four five")
        assert b.word_count == 5

    def test_word_count_empty_text(self):
        b = ModalityBlock(source="TYPED", text="", language="en")
        assert b.word_count == 0

    def test_all_valid_sources(self):
        for src in ("OCR", "HTR", "ASR", "TYPED"):
            b = ModalityBlock(source=src, text="hi", language="en")  # type: ignore[arg-type]
            assert b.source == src

    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            ModalityBlock(source="INVALID", text="hi", language="en")  # type: ignore[arg-type]

    def test_has_review_flags_default_false(self):
        b = make_block()
        assert b.has_review_flags is False


# ─── ReviewFlag ───────────────────────────────────────────────────────────────

class TestReviewFlag:
    def test_valid(self):
        f = ReviewFlag(modality="HTR", text="unclear handwriting", reason="low_confidence")
        assert f.modality == "HTR"

    def test_default_reason(self):
        f = ReviewFlag(modality="ASR", text="garbled")
        assert f.reason == "low_confidence"

    def test_invalid_modality(self):
        with pytest.raises(ValidationError):
            ReviewFlag(modality="INVALID", text="x")  # type: ignore[arg-type]


# ─── FusedDocument ────────────────────────────────────────────────────────────

class TestFusedDocument:
    def test_valid_construction(self):
        doc = make_doc()
        assert doc.primary_language == "en"
        assert len(doc.document_id) == 36  # UUID4

    def test_auto_total_word_count(self):
        doc = make_doc(fused_text="one two three four five six")
        assert doc.total_word_count == 6

    def test_auto_document_id_unique(self):
        d1 = make_doc()
        d2 = make_doc()
        assert d1.document_id != d2.document_id

    def test_has_content_true(self):
        doc = make_doc(fused_text="[OCR-DOCUMENT]\nSome text.\n[/OCR-DOCUMENT]")
        assert doc.has_content() is True

    def test_has_content_false_empty(self):
        doc = make_doc(fused_text="")
        assert doc.has_content() is False

    def test_has_content_false_whitespace(self):
        doc = make_doc(fused_text="   \n  ")
        assert doc.has_content() is False

    def test_modality_text_found(self):
        blocks = [
            make_block("OCR", "OCR text"),
            make_block("ASR", "ASR text"),
        ]
        doc = make_doc(modalities=blocks)
        assert doc.modality_text("OCR") == "OCR text"
        assert doc.modality_text("ASR") == "ASR text"

    def test_modality_text_not_found(self):
        doc = make_doc(modalities=[make_block("OCR", "text")])
        assert doc.modality_text("HTR") == ""

    def test_all_valid_intents(self):
        for intent in ("question", "instruction", "narration", "unknown"):
            doc = make_doc(intent=intent)
            assert doc.detected_intent == intent

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValidationError):
            make_doc(intent="gossip")

    def test_processing_time_non_negative(self):
        with pytest.raises(ValidationError):
            FusedDocument(
                modalities=[make_block()], fused_text="text",
                primary_language="en", detected_intent="unknown",
                processing_time_s=-0.1,
            )

    def test_serialisation_round_trip(self):
        flags = [ReviewFlag(modality="HTR", text="unclear")]
        doc = FusedDocument(
            modalities=[make_block("OCR", "text")],
            fused_text="[OCR-DOCUMENT]\ntext\n[/OCR-DOCUMENT]",
            primary_language="en", detected_intent="narration",
            review_flags=flags, processing_time_s=0.05,
        )
        data = doc.model_dump()
        restored = FusedDocument.model_validate(data)
        assert restored.document_id == doc.document_id
        assert len(restored.review_flags) == 1

    # ── for_rag() ─────────────────────────────────────────────────────────

    def test_for_rag_strips_ocr_tag(self):
        doc = make_doc(fused_text="[OCR-DOCUMENT]\nFiled under Section 138.\n[/OCR-DOCUMENT]")
        rag = doc.for_rag()
        assert "[OCR-DOCUMENT]" not in rag
        assert "[/OCR-DOCUMENT]" not in rag
        assert "Filed under Section 138." in rag

    def test_for_rag_strips_all_tags(self):
        text = (
            "[OCR-DOCUMENT]\nocr text\n[/OCR-DOCUMENT]\n\n"
            "[HTR-HANDWRITING]\nhtr text\n[/HTR-HANDWRITING]\n\n"
            "[ASR-VOICE]\nasr text\n[/ASR-VOICE]\n\n"
            "[TYPED-INPUT]\ntyped text\n[/TYPED-INPUT]"
        )
        doc = make_doc(fused_text=text)
        rag = doc.for_rag()
        for tag in ("OCR-DOCUMENT", "HTR-HANDWRITING", "ASR-VOICE", "TYPED-INPUT"):
            assert f"[{tag}]" not in rag
            assert f"[/{tag}]" not in rag

    def test_for_rag_preserves_review_markers(self):
        doc = make_doc(fused_text="[HTR-HANDWRITING]\n[REVIEW]unclear[/REVIEW]\n[/HTR-HANDWRITING]")
        rag = doc.for_rag()
        assert "[REVIEW]unclear[/REVIEW]" in rag

    def test_for_rag_plain_text_remains(self):
        doc = make_doc(fused_text="[OCR-DOCUMENT]\nImportant legal text.\n[/OCR-DOCUMENT]")
        rag = doc.for_rag()
        assert "Important legal text." in rag

    # ── for_llm() ─────────────────────────────────────────────────────────

    def test_for_llm_has_header(self):
        doc = make_doc()
        llm = doc.for_llm()
        assert "[LEGAL-AI CONTEXT]" in llm
        assert "[/LEGAL-AI CONTEXT]" in llm

    def test_for_llm_has_language(self):
        doc = make_doc()
        llm = doc.for_llm()
        assert "Language: en" in llm

    def test_for_llm_has_intent(self):
        doc = make_doc(intent="question")
        llm = doc.for_llm()
        assert "Intent: question" in llm

    def test_for_llm_contains_fused_text(self):
        ft = "[OCR-DOCUMENT]\nCritical clause text.\n[/OCR-DOCUMENT]"
        doc = make_doc(fused_text=ft)
        llm = doc.for_llm()
        assert "Critical clause text." in llm
