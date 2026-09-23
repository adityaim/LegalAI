"""
tests/asr/test_schema.py
─────────────────────────
Unit tests for the ASR output schema (Module 3).

Tests cover:
  • ASRSegment validation (required fields, bounds, Literal types)
  • Auto-derivation of review_required from confidence
  • end_s >= start_s enforcement
  • ASRResult construction and helper methods
  • Serialisation / round-trip (model_dump → model_validate)
  • to_fusion_text() [REVIEW] marker wrapping
  • logprob_to_confidence() mapping correctness
"""

import pytest
from pydantic import ValidationError

from modules.asr.postprocessor import logprob_to_confidence
from modules.asr.schema import ASRResult, ASRSegment


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_segment(
    segment_id: int = 0,
    start_s: float = 0.0,
    end_s: float = 5.0,
    text: str = "The accused appeared before the High Court.",
    confidence: float = 85.0,
    no_speech_prob: float = 0.02,
    asr_engine: str = "whisper",
    review_required: bool = False,
) -> ASRSegment:
    return ASRSegment(
        segment_id=segment_id,
        start_s=start_s,
        end_s=end_s,
        text=text,
        confidence=confidence,
        no_speech_prob=no_speech_prob,
        asr_engine=asr_engine,              # type: ignore[arg-type]
        review_required=review_required,
        language="en",
    )


def make_result(segments: list[ASRSegment] | None = None) -> ASRResult:
    segs = segments or [make_segment()]
    return ASRResult(
        source="ASR",
        language="en",
        language_probability=0.99,
        duration_s=10.0,
        transcript=" ".join(s.text for s in segs),
        segments=segs,
        intent="narration",
        processing_time_s=0.5,
        low_confidence_segments=sum(1 for s in segs if s.review_required),
    )


# ─── logprob_to_confidence tests ─────────────────────────────────────────────

class TestLogprobToConfidence:
    def test_zero_logprob_is_100(self):
        assert logprob_to_confidence(0.0) == pytest.approx(100.0)

    def test_minus4_logprob_is_0(self):
        assert logprob_to_confidence(-4.0) == pytest.approx(0.0)

    def test_minus1_is_75(self):
        assert logprob_to_confidence(-1.0) == pytest.approx(75.0)

    def test_below_minus4_clamps_to_0(self):
        assert logprob_to_confidence(-10.0) == 0.0

    def test_above_zero_clamps_to_100(self):
        assert logprob_to_confidence(0.5) == 100.0

    def test_threshold_at_minus1_2(self):
        # -1.2 logprob → exactly 70 confidence (review threshold)
        conf = logprob_to_confidence(-1.2)
        assert abs(conf - 70.0) < 0.01


# ─── ASRSegment tests ─────────────────────────────────────────────────────────

class TestASRSegment:
    def test_valid_construction(self):
        seg = make_segment()
        assert seg.text == "The accused appeared before the High Court."
        assert seg.confidence == 85.0
        assert seg.review_required is False

    def test_confidence_bounds_valid(self):
        make_segment(confidence=0.0)
        make_segment(confidence=100.0)

    def test_confidence_below_zero_invalid(self):
        with pytest.raises(ValidationError):
            make_segment(confidence=-1.0)

    def test_confidence_above_100_invalid(self):
        with pytest.raises(ValidationError):
            make_segment(confidence=101.0)

    def test_auto_review_low_confidence(self):
        seg = make_segment(confidence=40.0)
        assert seg.review_required is True

    def test_auto_review_exactly_threshold(self):
        seg = make_segment(confidence=70.0)
        assert seg.review_required is False  # 70.0 is NOT < 70

    def test_auto_review_just_below_threshold(self):
        seg = make_segment(confidence=69.9)
        assert seg.review_required is True

    def test_explicit_true_preserved(self):
        seg = make_segment(confidence=95.0, review_required=True)
        assert seg.review_required is True

    def test_end_before_start_invalid(self):
        with pytest.raises(ValidationError):
            ASRSegment(
                segment_id=0, start_s=5.0, end_s=3.0,
                text="invalid", confidence=80.0, asr_engine="whisper",
                no_speech_prob=0.1, language="en",
            )

    def test_end_equal_start_valid(self):
        seg = ASRSegment(
            segment_id=0, start_s=5.0, end_s=5.0,
            text="ok", confidence=80.0, asr_engine="whisper",
            no_speech_prob=0.1, language="en",
        )
        assert seg.end_s == seg.start_s

    def test_invalid_asr_engine_rejected(self):
        with pytest.raises(ValidationError):
            ASRSegment(
                segment_id=0, start_s=0.0, end_s=5.0,
                text="text", confidence=80.0,
                asr_engine="invalid_engine",  # type: ignore[arg-type]
                no_speech_prob=0.1, language="en",
            )

    def test_all_valid_asr_engines(self):
        for eng in ("whisper", "faster_whisper", "mock", "unknown"):
            seg = make_segment(asr_engine=eng)
            assert seg.asr_engine == eng

    def test_no_speech_prob_bounds(self):
        make_segment(no_speech_prob=0.0)
        make_segment(no_speech_prob=1.0)

    def test_no_speech_prob_above_1_invalid(self):
        with pytest.raises(ValidationError):
            make_segment(no_speech_prob=1.5)

    def test_serialisation_round_trip(self):
        seg = make_segment(confidence=55.0)
        data = seg.model_dump()
        restored = ASRSegment.model_validate(data)
        assert restored.text == seg.text
        assert restored.review_required is True
        assert restored.confidence == seg.confidence

    def test_segment_id_non_negative(self):
        with pytest.raises(ValidationError):
            make_segment(segment_id=-1)


# ─── ASRResult tests ─────────────────────────────────────────────────────────

class TestASRResult:
    def test_valid_construction(self):
        result = make_result()
        assert result.source == "ASR"
        assert result.language == "en"
        assert len(result.document_id) == 36  # UUID4

    def test_auto_document_id(self):
        r1 = make_result()
        r2 = make_result()
        assert r1.document_id != r2.document_id

    def test_mean_confidence_single(self):
        result = make_result([make_segment(confidence=80.0)])
        assert result.mean_confidence() == 80.0

    def test_mean_confidence_multiple(self):
        result = make_result([
            make_segment(confidence=80.0),
            make_segment(confidence=60.0),
        ])
        assert result.mean_confidence() == pytest.approx(70.0)

    def test_mean_confidence_empty(self):
        result = ASRResult(
            source="ASR", language="en", language_probability=1.0,
            duration_s=0.0, transcript="", segments=[],
            intent="unknown", processing_time_s=0.0, low_confidence_segments=0,
        )
        assert result.mean_confidence() == 0.0

    def test_flagged_segments(self):
        result = make_result([
            make_segment(confidence=90.0),
            make_segment(confidence=40.0),
            make_segment(confidence=75.0),
        ])
        flagged = result.flagged_segments()
        assert len(flagged) == 1
        assert flagged[0].confidence == 40.0

    def test_to_fusion_text_no_review(self):
        result = make_result([make_segment(confidence=90.0, text="Agreed.")])
        ft = result.to_fusion_text()
        assert "[REVIEW]" not in ft
        assert "Agreed." in ft

    def test_to_fusion_text_with_review(self):
        low = make_segment(confidence=40.0, text="garbled noise")
        result = make_result([low])
        ft = result.to_fusion_text()
        assert "[REVIEW]garbled noise[/REVIEW]" in ft

    def test_to_fusion_text_mixed(self):
        good = make_segment(segment_id=0, confidence=90.0, text="Good segment.")
        bad  = make_segment(segment_id=1, confidence=40.0, text="bad segment")
        result = make_result([good, bad])
        ft = result.to_fusion_text()
        assert "Good segment." in ft
        assert "[REVIEW]bad segment[/REVIEW]" in ft

    def test_serialisation_round_trip(self):
        result = make_result()
        data = result.model_dump()
        restored = ASRResult.model_validate(data)
        assert restored.document_id == result.document_id
        assert len(restored.segments) == len(result.segments)

    def test_processing_time_non_negative(self):
        with pytest.raises(ValidationError):
            ASRResult(
                source="ASR", language="en", language_probability=1.0,
                duration_s=0.0, transcript="", segments=[],
                intent="unknown", processing_time_s=-1.0, low_confidence_segments=0,
            )

    def test_all_valid_intents(self):
        for intent in ("question", "instruction", "narration", "unknown"):
            r = ASRResult(
                source="ASR", language="en", language_probability=1.0,
                duration_s=1.0, transcript="test", segments=[],
                intent=intent,                  # type: ignore[arg-type]
                processing_time_s=0.1, low_confidence_segments=0,
            )
            assert r.intent == intent

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValidationError):
            ASRResult(
                source="ASR", language="en", language_probability=1.0,
                duration_s=1.0, transcript="test", segments=[],
                intent="gossip",                # type: ignore[arg-type]
                processing_time_s=0.1, low_confidence_segments=0,
            )

    def test_language_probability_bounds(self):
        with pytest.raises(ValidationError):
            make_result().__class__(
                source="ASR", language="en", language_probability=1.5,
                duration_s=1.0, transcript="", segments=[],
                intent="unknown", processing_time_s=0.1, low_confidence_segments=0,
            )
