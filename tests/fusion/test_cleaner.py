"""
tests/fusion/test_cleaner.py
─────────────────────────────
Unit tests for the text cleaner (Module 4).

Tests cover:
  • clean_text() — NFKC, control chars, legal punct, whitespace
  • extract_review_flags() — span detection, multiple spans, empty
  • deduplicate_cross_modality() — exact matches, Jaccard near-dups,
    within-block no-dedup, single block passthrough
"""

import pytest

from modules.fusion.cleaner import (
    clean_text,
    deduplicate_cross_modality,
    extract_review_flags,
)


# ─── clean_text ───────────────────────────────────────────────────────────────

class TestCleanText:
    def test_empty_string(self):
        assert clean_text("") == ""

    def test_nkfc_ligature(self):
        # fi ligature → fi
        result = clean_text("\ufb01le")   # ﬁle → file
        assert result == "file"

    def test_control_chars_removed(self):
        result = clean_text("Hello\x00World\x01")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "HelloWorld" in result

    def test_newline_preserved(self):
        result = clean_text("Line one\nLine two")
        assert "\n" in result

    def test_multiple_spaces_collapsed(self):
        result = clean_text("Hello   World")
        assert "Hello World" in result

    def test_trailing_spaces_stripped(self):
        result = clean_text("  Hello  ")
        assert result == "Hello"

    def test_excessive_newlines_collapsed(self):
        result = clean_text("A\n\n\n\n\nB")
        assert "\n\n\n" not in result
        assert "A" in result and "B" in result

    def test_smart_quotes_normalised(self):
        result = clean_text("\u2018hello\u2019 and \u201cworld\u201d")
        assert "'" in result
        assert '"' in result

    def test_em_dash_normalised(self):
        result = clean_text("Alpha\u2014Beta")
        assert "-" in result

    def test_non_breaking_space_normalised(self):
        result = clean_text("Hello\u00a0World")
        assert "\u00a0" not in result

    def test_section_capitalised(self):
        result = clean_text("filed under section 138 of the act")
        assert "Section 138" in result

    def test_review_markers_preserved(self):
        result = clean_text("Text [REVIEW]unclear[/REVIEW] more text")
        assert "[REVIEW]unclear[/REVIEW]" in result

    def test_idempotent(self):
        text = "Section 138 of the NI Act applies here."
        assert clean_text(clean_text(text)) == clean_text(text)


# ─── extract_review_flags ─────────────────────────────────────────────────────

class TestExtractReviewFlags:
    def test_no_flags(self):
        flags = extract_review_flags("Normal text without review markers.", "OCR")
        assert flags == []

    def test_single_flag(self):
        flags = extract_review_flags("Text [REVIEW]unclear[/REVIEW] here.", "HTR")
        assert len(flags) == 1
        assert flags[0].text == "unclear"
        assert flags[0].modality == "HTR"
        assert flags[0].reason == "low_confidence"

    def test_multiple_flags(self):
        text = "[REVIEW]first[/REVIEW] normal [REVIEW]second[/REVIEW]"
        flags = extract_review_flags(text, "ASR")
        assert len(flags) == 2
        assert {f.text for f in flags} == {"first", "second"}

    def test_empty_review_span_skipped(self):
        flags = extract_review_flags("[REVIEW][/REVIEW]", "OCR")
        assert len(flags) == 0

    def test_multiline_span(self):
        text = "[REVIEW]line one\nline two[/REVIEW]"
        flags = extract_review_flags(text, "HTR")
        assert len(flags) == 1
        assert "line one" in flags[0].text

    def test_modality_preserved(self):
        for modality in ("OCR", "HTR", "ASR", "TYPED"):
            flags = extract_review_flags("[REVIEW]x[/REVIEW]", modality)  # type: ignore[arg-type]
            assert flags[0].modality == modality


# ─── deduplicate_cross_modality ───────────────────────────────────────────────

class TestDeduplicateCrossModality:
    def test_single_block_unchanged(self):
        blocks = [("OCR", "The plaintiff filed under Section 138.")]
        result = deduplicate_cross_modality(blocks)
        assert len(result) == 1
        assert "Section 138" in result[0]

    def test_exact_duplicate_removed_from_second(self):
        sentence = "The accused appeared before the High Court."
        blocks = [
            ("OCR", sentence),
            ("ASR", sentence),   # exact duplicate — should be dropped from ASR
        ]
        result = deduplicate_cross_modality(blocks)
        assert sentence in result[0]   # first block keeps it
        assert sentence not in result[1]  # second block loses it

    def test_near_duplicate_removed(self):
        # These two sentences share 9 of 10 unique words → Jaccard ≈ 0.90 > 0.85
        s1 = "The accused appeared before the High Court on the scheduled date."
        s2 = "The accused appeared before the High Court on scheduled date."
        blocks = [("OCR", s1), ("ASR", s2)]
        result = deduplicate_cross_modality(blocks)
        # First block keeps the sentence; second block should be empty or shorter
        assert s1 in result[0]
        # The near-dup should be removed from the second block
        assert result[1].strip() == "" or "accused appeared" not in result[1]

    def test_different_sentences_both_kept(self):
        s1 = "The contract was signed on 15 January 2025."
        s2 = "The accused failed to appear before the High Court."
        blocks = [("OCR", s1), ("ASR", s2)]
        result = deduplicate_cross_modality(blocks)
        assert s1 in result[0]
        assert s2 in result[1]

    def test_three_blocks_cascade_dedup(self):
        shared = "Alpha Technologies Pvt Ltd signed the agreement."
        blocks = [
            ("OCR",   shared),
            ("HTR",   shared),  # duplicate of OCR → removed
            ("TYPED", "What are the obligations?"),  # unique → kept
        ]
        result = deduplicate_cross_modality(blocks)
        assert shared in result[0]
        assert shared not in result[1]
        assert "obligations" in result[2]

    def test_empty_block(self):
        blocks = [("OCR", ""), ("ASR", "Important text.")]
        result = deduplicate_cross_modality(blocks)
        assert len(result) == 2
        assert "Important text." in result[1]

    def test_two_unique_blocks_unchanged(self):
        s1 = "Clause 1: Payment within 30 days."
        s2 = "The High Court adjourned the matter to February."
        blocks = [("OCR", s1), ("ASR", s2)]
        result = deduplicate_cross_modality(blocks)
        assert s1 in result[0]
        assert s2 in result[1]
