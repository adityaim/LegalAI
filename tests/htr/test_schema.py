"""
tests/htr/test_schema.py
─────────────────────────
Unit tests for the HTR output schema (Module 2).

Tests cover:
  • HTRRegion validation (required fields, bounds, type literals)
  • Auto-derivation of review_required from confidence
  • HTRResult construction and helper methods
  • Serialisation / round-trip (model_dump → model_validate)
  • to_fusion_text() [REVIEW] marker wrapping
"""

import pytest
from pydantic import ValidationError

from modules.htr.schema import HTRRegion, HTRResult
from modules.ocr.schema import BoundingBox


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_bbox(page: int = 1) -> BoundingBox:
    return BoundingBox(x0=10.0, y0=20.0, x1=200.0, y1=80.0, page=page)


def make_region(
    confidence: float = 90.0,
    text: str = "Signed by counsel",
    hw_type: str = "signature",
    page: int = 1,
    review_required: bool = False,
    htr_engine: str = "trocr",
) -> HTRRegion:
    return HTRRegion(
        page=page,
        type=hw_type,       # type: ignore[arg-type]
        text=text,
        confidence=confidence,
        bbox=make_bbox(page=page),
        review_required=review_required,
        htr_engine=htr_engine,  # type: ignore[arg-type]
        language="en",
    )


def make_result(regions: list[HTRRegion] | None = None) -> HTRResult:
    regs = regions or [make_region()]
    return HTRResult(
        source="HTR",
        language="en",
        regions=regs,
        full_text=" ".join(r.text for r in regs),
        processing_time_s=0.5,
        low_confidence_regions=sum(1 for r in regs if r.review_required),
    )


# ─── HTRRegion tests ─────────────────────────────────────────────────────────

class TestHTRRegion:
    def test_valid_construction(self):
        r = make_region()
        assert r.text == "Signed by counsel"
        assert r.confidence == 90.0
        assert r.review_required is False

    def test_confidence_bounds_valid(self):
        make_region(confidence=0.0)
        make_region(confidence=100.0)

    def test_confidence_below_zero_invalid(self):
        with pytest.raises(ValidationError):
            make_region(confidence=-1.0)

    def test_confidence_above_100_invalid(self):
        with pytest.raises(ValidationError):
            make_region(confidence=101.0)

    def test_auto_review_low_confidence(self):
        r = make_region(confidence=40.0)
        assert r.review_required is True

    def test_auto_review_exactly_threshold(self):
        r = make_region(confidence=70.0)
        assert r.review_required is False   # 70.0 is NOT < 70

    def test_auto_review_just_below_threshold(self):
        r = make_region(confidence=69.9)
        assert r.review_required is True

    def test_explicit_true_preserved(self):
        r = make_region(confidence=95.0, review_required=True)
        assert r.review_required is True

    def test_invalid_hw_type(self):
        with pytest.raises(ValidationError):
            HTRRegion(
                page=1, type="invalid_type", text="x", confidence=80.0,
                bbox=make_bbox(), htr_engine="trocr", language="en",
            )

    def test_all_valid_hw_types(self):
        for hw_type in ("standalone_note", "annotation", "signature", "table_fill", "unknown"):
            r = make_region(hw_type=hw_type)
            assert r.type == hw_type

    def test_invalid_htr_engine_rejected(self):
        with pytest.raises(ValidationError):
            HTRRegion(
                page=1, type="signature", text="x", confidence=80.0,
                bbox=make_bbox(), htr_engine="invalid_engine", language="en",
            )

    def test_page_zero_invalid(self):
        with pytest.raises(ValidationError):
            HTRRegion(
                page=0, type="signature", text="x", confidence=80.0,
                bbox=make_bbox(), htr_engine="trocr", language="en",
            )

    def test_serialisation_round_trip(self):
        r = make_region(confidence=55.0)
        data = r.model_dump()
        restored = HTRRegion.model_validate(data)
        assert restored.text == r.text
        assert restored.review_required is True
        assert restored.confidence == r.confidence


# ─── HTRResult tests ─────────────────────────────────────────────────────────

class TestHTRResult:
    def test_valid_construction(self):
        result = make_result()
        assert result.source == "HTR"
        assert result.language == "en"
        assert len(result.document_id) == 36  # UUID4

    def test_auto_document_id(self):
        r1 = make_result()
        r2 = make_result()
        assert r1.document_id != r2.document_id  # UUIDs are unique

    def test_mean_confidence_single(self):
        result = make_result([make_region(confidence=80.0)])
        assert result.mean_confidence() == 80.0

    def test_mean_confidence_multiple(self):
        result = make_result([
            make_region(confidence=80.0),
            make_region(confidence=60.0),
        ])
        assert result.mean_confidence() == pytest.approx(70.0)

    def test_mean_confidence_empty(self):
        result = HTRResult(
            source="HTR", language="en", regions=[],
            full_text="", processing_time_s=0.0, low_confidence_regions=0,
        )
        assert result.mean_confidence() == 0.0

    def test_flagged_regions(self):
        result = make_result([
            make_region(confidence=90.0),
            make_region(confidence=40.0),
            make_region(confidence=75.0),
        ])
        flagged = result.flagged_regions()
        assert len(flagged) == 1
        assert flagged[0].confidence == 40.0

    def test_regions_by_type(self):
        result = make_result([
            make_region(hw_type="signature"),
            make_region(hw_type="annotation"),
            make_region(hw_type="signature"),
        ])
        sigs = result.regions_by_type("signature")
        assert len(sigs) == 2

    def test_to_fusion_text_no_review(self):
        result = make_result([make_region(confidence=90.0, text="Agreed")])
        ft = result.to_fusion_text()
        assert "[REVIEW]" not in ft
        assert "Agreed" in ft

    def test_to_fusion_text_with_review(self):
        low = make_region(confidence=40.0, text="scribble")
        result = make_result([low])
        ft = result.to_fusion_text()
        assert "[REVIEW]scribble[/REVIEW]" in ft

    def test_serialisation_round_trip(self):
        result = make_result()
        data = result.model_dump()
        restored = HTRResult.model_validate(data)
        assert restored.document_id == result.document_id
        assert len(restored.regions) == len(result.regions)

    def test_processing_time_non_negative(self):
        with pytest.raises(ValidationError):
            make_result().__class__(
                source="HTR", language="en", regions=[], full_text="",
                processing_time_s=-1.0, low_confidence_regions=0,
            )
