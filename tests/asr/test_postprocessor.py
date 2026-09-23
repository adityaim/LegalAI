"""
tests/asr/test_postprocessor.py
─────────────────────────────────
Unit tests for the ASR post-processor (Module 3).

Tests cover:
  • logprob_to_confidence() mapping
  • Silence segment filtering (no_speech_prob threshold)
  • Legal term normalisation (IPC, Section 138, NI Act, etc.)
  • Whitespace cleaning
  • Transcript assembly with [REVIEW] markers
  • Intent detection (question / instruction / narration / unknown)
"""

import pytest

from modules.asr.engines.base import SegmentResult
from modules.asr.postprocessor import (
    ASRPostprocessor,
    _apply_legal_corrections,
    detect_intent,
    logprob_to_confidence,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_raw(
    segment_id: int = 0,
    start_s: float = 0.0,
    end_s: float = 5.0,
    text: str = "The plaintiff filed under Section 138.",
    avg_logprob: float = -0.5,
    no_speech_prob: float = 0.02,
    language: str = "en",
) -> SegmentResult:
    return SegmentResult(
        segment_id=segment_id,
        start_s=start_s,
        end_s=end_s,
        text=text,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        language=language,
    )


# ─── logprob_to_confidence ────────────────────────────────────────────────────

class TestLogprobToConfidence:
    def test_zero_maps_to_100(self):
        assert logprob_to_confidence(0.0) == 100.0

    def test_minus4_maps_to_0(self):
        assert logprob_to_confidence(-4.0) == 0.0

    def test_minus1_maps_to_75(self):
        assert logprob_to_confidence(-1.0) == pytest.approx(75.0)

    def test_clamped_below_0(self):
        assert logprob_to_confidence(-100.0) == 0.0

    def test_clamped_above_100(self):
        assert logprob_to_confidence(0.1) == 100.0

    def test_review_threshold_at_minus1_2(self):
        conf = logprob_to_confidence(-1.2)
        assert abs(conf - 70.0) < 0.1


# ─── Legal corrections ────────────────────────────────────────────────────────

class TestLegalCorrections:
    def test_section_138_space(self):
        assert "Section 138" in _apply_legal_corrections("section 1 38 of the act")

    def test_ipc_spaced(self):
        assert "IPC" in _apply_legal_corrections("under i p c")

    def test_crpc_spaced(self):
        assert "CrPC" in _apply_legal_corrections("c r p c applies")

    def test_ni_act(self):
        result = _apply_legal_corrections("under n.i.act provisions")
        assert "NI Act" in result

    def test_high_court_capitalisation(self):
        assert "High Court" in _apply_legal_corrections("appeared before high court")

    def test_supreme_court_capitalisation(self):
        assert "Supreme Court" in _apply_legal_corrections("appealed to supreme court")

    def test_rera_spaced(self):
        assert "RERA" in _apply_legal_corrections("r e r a complaint")

    def test_no_change_when_correct(self):
        text = "Filed under Section 138 of the NI Act."
        assert _apply_legal_corrections(text) == text

    def test_double_space_section(self):
        result = _apply_legal_corrections("Section  302")
        assert "Section  302" not in result
        assert "Section 302" in result


# ─── Intent detection ─────────────────────────────────────────────────────────

class TestDetectIntent:
    def test_question_what(self):
        assert detect_intent("What does Section 138 say?") == "question"

    def test_question_how(self):
        assert detect_intent("How does the court proceed in this case?") == "question"

    def test_question_mark_alone(self):
        assert detect_intent("Is this admissible?") == "question"

    def test_instruction_draft(self):
        assert detect_intent("Draft a legal notice under IPC Section 420.") == "instruction"

    def test_instruction_file(self):
        assert detect_intent("File the petition by Friday.") == "instruction"

    def test_narration(self):
        assert detect_intent(
            "The accused failed to appear before the court on the scheduled date."
        ) == "narration"

    def test_empty_is_unknown(self):
        assert detect_intent("") == "unknown"

    def test_whitespace_is_unknown(self):
        assert detect_intent("   ") == "unknown"

    def test_question_priority_over_instruction(self):
        # "How do I draft" — question wins
        assert detect_intent("How do I draft a petition?") == "question"


# ─── ASRPostprocessor ─────────────────────────────────────────────────────────

class TestASRPostprocessor:
    def setup_method(self):
        self.pp = ASRPostprocessor()

    def test_basic_segment_conversion(self):
        raw = [make_raw(avg_logprob=-0.5)]
        segs = self.pp.process_segments(raw, engine_tag="whisper")
        assert len(segs) == 1
        assert segs[0].asr_engine == "whisper"
        expected_conf = logprob_to_confidence(-0.5)
        assert segs[0].confidence == pytest.approx(expected_conf)

    def test_silence_dropped(self):
        raw = [
            make_raw(text="good", no_speech_prob=0.05),
            make_raw(text="silence", no_speech_prob=0.9),   # should be dropped
        ]
        segs = self.pp.process_segments(raw)
        assert len(segs) == 1
        assert segs[0].text == "good"

    def test_silence_at_threshold_kept(self):
        # no_speech_prob == 0.6 is NOT > threshold (strict >), so kept
        raw = [make_raw(text="borderline", no_speech_prob=0.6)]
        segs = self.pp.process_segments(raw)
        assert len(segs) == 1

    def test_empty_text_dropped(self):
        raw = [make_raw(text="   ")]
        segs = self.pp.process_segments(raw)
        assert len(segs) == 0

    def test_low_logprob_sets_review_required(self):
        raw = [make_raw(avg_logprob=-3.5)]  # confidence ≈ 12.5 < 70
        segs = self.pp.process_segments(raw)
        assert segs[0].review_required is True

    def test_high_logprob_no_review(self):
        raw = [make_raw(avg_logprob=-0.2)]  # confidence ≈ 95 > 70
        segs = self.pp.process_segments(raw)
        assert segs[0].review_required is False

    def test_legal_corrections_applied(self):
        raw = [make_raw(text="filed under i p c section 138")]
        segs = self.pp.process_segments(raw, engine_tag="whisper")
        assert "IPC" in segs[0].text
        assert "Section 138" in segs[0].text

    def test_legal_corrections_disabled(self):
        pp = ASRPostprocessor(enable_legal_corrections=False)
        raw = [make_raw(text="i p c case")]
        segs = pp.process_segments(raw)
        assert "IPC" not in segs[0].text

    def test_unknown_engine_tag_mapped(self):
        raw = [make_raw()]
        segs = self.pp.process_segments(raw, engine_tag="nonexistent_engine")
        assert segs[0].asr_engine == "unknown"

    def test_assemble_transcript_no_review(self):
        segs = self.pp.process_segments([
            make_raw(text="Section 138 case.", avg_logprob=-0.3),
        ])
        transcript = self.pp.assemble_transcript(segs)
        assert "[REVIEW]" not in transcript
        assert "Section 138 case." in transcript

    def test_assemble_transcript_with_review(self):
        segs = self.pp.process_segments([
            make_raw(text="garbled text", avg_logprob=-3.8),
        ])
        transcript = self.pp.assemble_transcript(segs)
        assert "[REVIEW]garbled text[/REVIEW]" in transcript

    def test_assemble_transcript_multiple_segments(self):
        raw = [
            make_raw(segment_id=0, text="First segment.", avg_logprob=-0.2),
            make_raw(segment_id=1, text="Second segment.", avg_logprob=-0.4),
        ]
        segs = self.pp.process_segments(raw)
        transcript = self.pp.assemble_transcript(segs)
        assert "First segment." in transcript
        assert "Second segment." in transcript

    def test_detect_intent_question(self):
        intent = self.pp.detect_intent("What does the NI Act say?")
        assert intent == "question"

    def test_detect_intent_instruction(self):
        intent = self.pp.detect_intent("Draft a notice for the respondent.")
        assert intent == "instruction"

    def test_detect_intent_narration(self):
        intent = self.pp.detect_intent("The court adjourned the matter.")
        assert intent == "narration"

    def test_whitespace_cleaning(self):
        raw = [make_raw(text="  Section  138  ")]
        segs = self.pp.process_segments(raw)
        assert not segs[0].text.startswith(" ")
        assert not segs[0].text.endswith(" ")

    def test_multiple_silence_segments_all_dropped(self):
        raw = [
            make_raw(segment_id=i, text=f"seg{i}", no_speech_prob=0.95)
            for i in range(5)
        ]
        segs = self.pp.process_segments(raw)
        assert len(segs) == 0
