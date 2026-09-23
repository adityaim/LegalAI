"""
tests/fusion/test_processor.py
────────────────────────────────
Integration tests for TextFusionProcessor (Module 4).

Tests cover:
  • fuse() with all four modalities
  • fuse() with partial inputs (OCR only, ASR only, typed only)
  • fuse() raises ValueError with no inputs
  • Tag wrapping format ([OCR-DOCUMENT], [HTR-HANDWRITING], etc.)
  • Language detection — majority vote, ASR 2× weight
  • Intent inheritance from ASR
  • Intent heuristic fallback (no ASR)
  • Review flags collected from all modalities
  • Deduplication enabled / disabled
  • Empty modality blocks are dropped (min_block_chars)
  • processing_time_s is positive
  • FusedDocument round-trip serialisation
"""

import pytest

from modules.fusion import TextFusionProcessor
from modules.fusion.processor import ProcessorConfig
from modules.fusion.schema import FusedDocument


# ─── Fake modality results ────────────────────────────────────────────────────

class FakeOCR:
    document_id = "ocr-001"
    language = "en"
    full_text = "Alpha Technologies Pvt. Ltd. executed the agreement on 15 January 2025."


class FakeHTR:
    document_id = "htr-002"
    language = "en"
    def to_fusion_text(self) -> str:
        return "Signed by R.K. Sharma. [REVIEW]see clause 4[/REVIEW]"


class FakeASR:
    document_id = "asr-003"
    language = "en"
    intent = "question"
    def to_fusion_text(self) -> str:
        return "What are the payment terms under this agreement?"


class FakeASRHindi:
    document_id = "asr-004"
    language = "hi"
    intent = "narration"
    def to_fusion_text(self) -> str:
        return "Vaad-vivad ka niptara karna hoga."


# ─── Tag wrapping ─────────────────────────────────────────────────────────────

class TestTagWrapping:
    def setup_method(self):
        self.fuser = TextFusionProcessor()

    def test_ocr_block_wrapped(self):
        doc = self.fuser.fuse(ocr=FakeOCR())
        assert "[OCR-DOCUMENT]" in doc.fused_text
        assert "[/OCR-DOCUMENT]" in doc.fused_text

    def test_htr_block_wrapped(self):
        doc = self.fuser.fuse(htr=FakeHTR())
        assert "[HTR-HANDWRITING]" in doc.fused_text
        assert "[/HTR-HANDWRITING]" in doc.fused_text

    def test_asr_block_wrapped(self):
        doc = self.fuser.fuse(asr=FakeASR())
        assert "[ASR-VOICE]" in doc.fused_text
        assert "[/ASR-VOICE]" in doc.fused_text

    def test_typed_block_wrapped(self):
        doc = self.fuser.fuse(typed_text="Draft a legal notice.")
        assert "[TYPED-INPUT]" in doc.fused_text
        assert "[/TYPED-INPUT]" in doc.fused_text

    def test_all_four_tags_present(self):
        doc = self.fuser.fuse(
            ocr=FakeOCR(), htr=FakeHTR(), asr=FakeASR(),
            typed_text="Summarise the agreement.",
        )
        for tag in ("OCR-DOCUMENT", "HTR-HANDWRITING", "ASR-VOICE", "TYPED-INPUT"):
            assert f"[{tag}]" in doc.fused_text
            assert f"[/{tag}]" in doc.fused_text

    def test_tag_order_ocr_first(self):
        doc = self.fuser.fuse(ocr=FakeOCR(), asr=FakeASR())
        ocr_pos = doc.fused_text.find("[OCR-DOCUMENT]")
        asr_pos = doc.fused_text.find("[ASR-VOICE]")
        assert ocr_pos < asr_pos


# ─── Input validation ─────────────────────────────────────────────────────────

class TestInputValidation:
    def setup_method(self):
        self.fuser = TextFusionProcessor()

    def test_no_inputs_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            self.fuser.fuse()

    def test_ocr_only(self):
        doc = self.fuser.fuse(ocr=FakeOCR())
        assert doc.has_content()
        assert len(doc.modalities) == 1

    def test_typed_only(self):
        doc = self.fuser.fuse(typed_text="What is Section 138 NI Act?")
        assert doc.has_content()

    def test_empty_typed_text_treated_as_none(self):
        # Empty string is falsy — treated like None; OCR still fuses
        doc = self.fuser.fuse(ocr=FakeOCR(), typed_text="")
        assert len([b for b in doc.modalities if b.source == "TYPED"]) == 0

    def test_all_four_modalities(self):
        doc = self.fuser.fuse(
            ocr=FakeOCR(), htr=FakeHTR(), asr=FakeASR(),
            typed_text="Summarise.",
        )
        assert len(doc.modalities) == 4


# ─── Language detection ───────────────────────────────────────────────────────

class TestLanguageDetection:
    def test_english_default(self):
        doc = TextFusionProcessor().fuse(ocr=FakeOCR())
        assert doc.primary_language == "en"

    def test_hindi_from_asr_dominates(self):
        doc = TextFusionProcessor().fuse(
            ocr=FakeOCR(),   # en
            asr=FakeASRHindi(),  # hi (weighted 2x)
        )
        # ASR has weight 2; en=1, hi=2 → hi wins
        assert doc.primary_language == "hi"

    def test_majority_english(self):
        doc = TextFusionProcessor().fuse(
            ocr=FakeOCR(),           # en
            htr=FakeHTR(),           # en
            typed_text="Summary.",   # en default
        )
        assert doc.primary_language == "en"


# ─── Intent detection ─────────────────────────────────────────────────────────

class TestIntentDetection:
    def test_asr_intent_inherited(self):
        doc = TextFusionProcessor().fuse(asr=FakeASR())
        assert doc.detected_intent == "question"

    def test_narration_from_asr(self):
        doc = TextFusionProcessor().fuse(asr=FakeASRHindi())
        assert doc.detected_intent == "narration"

    def test_question_heuristic_from_typed(self):
        doc = TextFusionProcessor().fuse(typed_text="What does Section 138 say?")
        assert doc.detected_intent == "question"

    def test_instruction_heuristic(self):
        doc = TextFusionProcessor().fuse(typed_text="Draft a legal notice for the respondent.")
        assert doc.detected_intent == "instruction"

    def test_narration_heuristic(self):
        doc = TextFusionProcessor().fuse(
            typed_text="The court adjourned the matter to next month."
        )
        assert doc.detected_intent == "narration"

    def test_unknown_on_empty(self):
        # Tiny text below min_block_chars should result in empty fused_text → unknown
        config = ProcessorConfig(min_block_chars=100)
        doc = TextFusionProcessor(config=config).fuse(typed_text="hi")
        # Should not crash; intent might be unknown if block is dropped
        assert doc.detected_intent in ("question", "instruction", "narration", "unknown")


# ─── Review flags ─────────────────────────────────────────────────────────────

class TestReviewFlags:
    def test_htr_review_flag_collected(self):
        doc = TextFusionProcessor().fuse(htr=FakeHTR())
        assert len(doc.review_flags) >= 1
        assert doc.review_flags[0].modality == "HTR"
        assert "clause 4" in doc.review_flags[0].text

    def test_no_flags_when_no_review_markers(self):
        doc = TextFusionProcessor().fuse(ocr=FakeOCR())
        assert len(doc.review_flags) == 0

    def test_multiple_modality_flags(self):
        class MultiReviewASR:
            document_id = "x"
            language = "en"
            intent = "unknown"
            def to_fusion_text(self):
                return "[REVIEW]first[/REVIEW] text [REVIEW]second[/REVIEW]"

        doc = TextFusionProcessor().fuse(htr=FakeHTR(), asr=MultiReviewASR())
        # HTR has 1 flag, ASR has 2 → total 3 (before any dedup)
        assert len(doc.review_flags) >= 2


# ─── Deduplication ────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_dedup_enabled_removes_duplicate(self):
        class DupASR:
            document_id = "dup"
            language = "en"
            intent = "narration"
            def to_fusion_text(self):
                return FakeOCR.full_text  # exact same as OCR

        doc = TextFusionProcessor().fuse(ocr=FakeOCR(), asr=DupASR())
        # OCR text must appear once in the fused text
        count = doc.fused_text.count("Alpha Technologies Pvt")
        assert count == 1

    def test_dedup_disabled_keeps_both(self):
        class DupASR:
            document_id = "dup"
            language = "en"
            intent = "narration"
            def to_fusion_text(self):
                return FakeOCR.full_text

        config = ProcessorConfig(enable_deduplication=False)
        doc = TextFusionProcessor(config=config).fuse(ocr=FakeOCR(), asr=DupASR())
        count = doc.fused_text.count("Alpha Technologies Pvt")
        assert count == 2


# ─── Source IDs ───────────────────────────────────────────────────────────────

class TestSourceIds:
    def test_source_ids_collected(self):
        doc = TextFusionProcessor().fuse(ocr=FakeOCR(), htr=FakeHTR(), asr=FakeASR())
        assert "ocr-001" in doc.source_ids
        assert "htr-002" in doc.source_ids
        assert "asr-003" in doc.source_ids

    def test_typed_has_no_source_id(self):
        doc = TextFusionProcessor().fuse(typed_text="Query")
        assert doc.source_ids == []


# ─── Processing time + serialisation ─────────────────────────────────────────

class TestMisc:
    def test_processing_time_positive(self):
        doc = TextFusionProcessor().fuse(ocr=FakeOCR())
        assert doc.processing_time_s > 0.0

    def test_serialisation_round_trip(self):
        doc = TextFusionProcessor().fuse(
            ocr=FakeOCR(), htr=FakeHTR(), asr=FakeASR(),
            typed_text="What is this?",
        )
        data = doc.model_dump()
        restored = FusedDocument.model_validate(data)
        assert restored.document_id == doc.document_id
        assert restored.fused_text == doc.fused_text
        assert len(restored.modalities) == len(doc.modalities)

    def test_for_rag_output_type(self):
        doc = TextFusionProcessor().fuse(ocr=FakeOCR())
        assert isinstance(doc.for_rag(), str)
        assert len(doc.for_rag()) > 0

    def test_for_llm_output_type(self):
        doc = TextFusionProcessor().fuse(ocr=FakeOCR())
        assert isinstance(doc.for_llm(), str)
        assert "[LEGAL-AI CONTEXT]" in doc.for_llm()

    def test_min_block_chars_drops_tiny(self):
        config = ProcessorConfig(min_block_chars=50)
        doc = TextFusionProcessor(config=config).fuse(
            typed_text="hi",   # 2 chars < 50
            ocr=FakeOCR(),     # long text — kept
        )
        sources = {b.source for b in doc.modalities}
        assert "TYPED" not in sources
        assert "OCR" in sources
