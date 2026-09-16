"""
tests/htr/test_postprocessor.py
─────────────────────────────────
Unit tests for the HTR post-processor.

Tests cover:
  • SymSpell corrector no-ops when unavailable (graceful degradation)
  • Legal term preservation (IPC, Section 138, AIR citations not corrected)
  • _clean() removes control chars, collapses whitespace
  • process_regions() sets review_required correctly
  • assemble_text() wraps low-confidence regions in [REVIEW] markers
  • assemble_text() skips empty region texts
"""

import pytest

from modules.htr.postprocessor import HTRPostprocessor, SymSpellCorrector, _has_legal_token
from modules.htr.schema import HTRRegion
from modules.ocr.schema import BoundingBox


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_region(text: str, confidence: float, page: int = 1) -> HTRRegion:
    return HTRRegion(
        page=page,
        type="standalone_note",
        text=text,
        confidence=confidence,
        bbox=BoundingBox(x0=10.0, y0=10.0, x1=200.0, y1=50.0, page=page),
        review_required=False,
        htr_engine="trocr",
        language="en",
    )


# ─── Legal term preservation ──────────────────────────────────────────────────

class TestLegalTermPreservation:
    def test_ipc_preserved(self):
        assert _has_legal_token("under IPC") is True

    def test_crpc_preserved(self):
        assert _has_legal_token("CrPC section") is True

    def test_section_ref_preserved(self):
        assert _has_legal_token("Section 138") is True

    def test_section_abbrev_preserved(self):
        assert _has_legal_token("S. 302 IPC") is True

    def test_air_citation_preserved(self):
        assert _has_legal_token("2021 AIR 100") is True

    def test_scc_citation_preserved(self):
        assert _has_legal_token("2024 SCC 56") is True

    def test_all_caps_preserved(self):
        assert _has_legal_token("RERA") is True

    def test_number_preserved(self):
        assert _has_legal_token("12345") is True

    def test_plain_text_not_preserved(self):
        assert _has_legal_token("hereby agrees") is False


# ─── Text cleaning ────────────────────────────────────────────────────────────

class TestClean:
    def setup_method(self):
        self.pp = HTRPostprocessor(enable_spellcheck=False)

    def test_control_chars_removed(self):
        cleaned = self.pp._clean("text\x00with\x07null")
        assert "\x00" not in cleaned
        assert "\x07" not in cleaned

    def test_whitespace_collapsed(self):
        cleaned = self.pp._clean("too   many    spaces")
        assert "  " not in cleaned

    def test_newlines_preserved(self):
        cleaned = self.pp._clean("line1\nline2")
        assert "\n" in cleaned

    def test_excess_newlines_collapsed(self):
        cleaned = self.pp._clean("line1\n\n\n\nline2")
        assert cleaned.count("\n") <= 3

    def test_strip(self):
        cleaned = self.pp._clean("   hello   ")
        assert cleaned == "hello"


# ─── process_regions() ────────────────────────────────────────────────────────

class TestProcessRegions:
    def setup_method(self):
        self.pp = HTRPostprocessor(enable_spellcheck=False)

    def test_high_confidence_not_flagged(self):
        regions = self.pp.process_regions([make_region("text", 90.0)])
        assert regions[0].review_required is False

    def test_low_confidence_flagged(self):
        regions = self.pp.process_regions([make_region("text", 50.0)])
        assert regions[0].review_required is True

    def test_exactly_at_threshold_not_flagged(self):
        regions = self.pp.process_regions([make_region("text", 70.0)])
        assert regions[0].review_required is False

    def test_just_below_threshold_flagged(self):
        regions = self.pp.process_regions([make_region("text", 69.9)])
        assert regions[0].review_required is True

    def test_mixed_regions(self):
        regions = self.pp.process_regions([
            make_region("good text", 95.0),
            make_region("bad text", 40.0),
            make_region("ok text", 75.0),
        ])
        assert regions[0].review_required is False
        assert regions[1].review_required is True
        assert regions[2].review_required is False

    def test_text_cleaned(self):
        regions = self.pp.process_regions([make_region("hello\x00world", 90.0)])
        assert "\x00" not in regions[0].text

    def test_empty_regions_preserved(self):
        regions = self.pp.process_regions([make_region("", 30.0)])
        assert regions[0].text == ""
        assert regions[0].review_required is True


# ─── assemble_text() ─────────────────────────────────────────────────────────

class TestAssembleText:
    def setup_method(self):
        self.pp = HTRPostprocessor(enable_spellcheck=False)

    def _process(self, regions: list[HTRRegion]) -> str:
        processed = self.pp.process_regions(regions)
        return self.pp.assemble_text(processed)

    def test_no_review_no_markers(self):
        text = self._process([make_region("Agreed", 90.0)])
        assert "[REVIEW]" not in text
        assert "Agreed" in text

    def test_low_conf_wrapped_in_markers(self):
        text = self._process([make_region("scribble", 40.0)])
        assert "[REVIEW]scribble[/REVIEW]" in text

    def test_empty_regions_skipped(self):
        text = self._process([
            make_region("", 90.0),
            make_region("Real text", 90.0),
        ])
        assert "Real text" in text

    def test_multiple_regions_joined(self):
        text = self._process([
            make_region("first note", 90.0),
            make_region("second note", 90.0),
        ])
        assert "first note" in text
        assert "second note" in text

    def test_empty_input(self):
        text = self._process([])
        assert text == ""

    def test_mixed_confidence_markers(self):
        text = self._process([
            make_region("clear text", 85.0),
            make_region("unclear", 35.0),
            make_region("clear again", 92.0),
        ])
        assert "[REVIEW]unclear[/REVIEW]" in text
        assert "[REVIEW]clear text[/REVIEW]" not in text
        assert "[REVIEW]clear again[/REVIEW]" not in text


# ─── SymSpell availability (smoke test) ──────────────────────────────────────

class TestSymSpellCorrector:
    def test_correct_returns_string(self):
        corrector = SymSpellCorrector()
        result = corrector.correct("hereby")
        assert isinstance(result, str)

    def test_empty_string(self):
        corrector = SymSpellCorrector()
        result = corrector.correct("")
        assert result == ""

    def test_legal_token_not_corrupted(self):
        """Legal tokens must pass through unchanged."""
        corrector = SymSpellCorrector()
        # IPC is all-caps → preserved by regex check before SymSpell
        result = corrector.correct("under IPC Section 138")
        assert "IPC" in result
        assert "138" in result
