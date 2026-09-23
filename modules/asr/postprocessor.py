"""
modules/asr/postprocessor.py
──────────────────────────────
Post-processing for ASR output (Module 3).

Responsibilities:
  1. Confidence mapping — convert Whisper avg_logprob (−∞, 0] → confidence 0–100
     Formula: confidence = max(0, min(100, (1 + avg_logprob / 4) * 100))
     Chosen so that:
       avg_logprob = 0   → confidence = 100  (perfect)
       avg_logprob = -1  → confidence = 75   (good, above threshold)
       avg_logprob = -1.2→ confidence = 70   (review threshold)
       avg_logprob = -4  → confidence = 0    (garbage)

  2. Silence filtering — drop segments where no_speech_prob > 0.6

  3. Legal term normalisation — fix common Whisper mis-transcriptions
     of Indian legal terms (e.g. "IBC" confused with "EBC",
     "Section 138" written as "section 1 38", "SC" as "es see").

  4. Transcript assembly — join segments in order, [REVIEW] marker wrapping.

  5. Intent detection — classify the utterance as question / instruction /
     narration / unknown using simple keyword heuristics.
"""

from __future__ import annotations

import logging
import re

from .engines.base import SegmentResult
from .schema import ASRSegment, LegalIntent

logger = logging.getLogger(__name__)

# ─── Confidence threshold ─────────────────────────────────────────────────────

LOW_CONFIDENCE_THRESHOLD = 70.0
NO_SPEECH_THRESHOLD      = 0.6   # drop segments with higher no_speech_prob


# ─── avg_logprob → confidence mapping ────────────────────────────────────────

def logprob_to_confidence(avg_logprob: float) -> float:
    """
    Map Whisper avg_logprob (−∞, 0] to confidence 0–100.

    Formula: max(0, min(100, (1 + avg_logprob / 4) * 100))

    Key points:
      0   → 100%   (perfect decoding)
      -1  → 75%    (good)
      -1.2→ 70%    (exactly at review threshold)
      -4  → 0%     (total failure)

    Reference: derived from faster-whisper's log_prob_threshold=-1.0 default
    which is the common cut-off for reliable transcription.
    """
    return max(0.0, min(100.0, (1.0 + avg_logprob / 4.0) * 100.0))


# ─── Legal term normalisation patterns ────────────────────────────────────────

# Common Whisper mis-transcriptions of Indian legal vocabulary.
# Format: (compiled_regex, replacement_string)
_LEGAL_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # "section 1 38" / "section one thirty eight" → "Section 138"
    (re.compile(r"\bsection\s+1\s+38\b", re.IGNORECASE), "Section 138"),
    (re.compile(r"\bsection\s+one\s+thirty[\s-]eight\b", re.IGNORECASE), "Section 138"),
    # Capitalise lowercase "section NNN" → "Section NNN"
    (re.compile(r"\bsection\s+(\d+)\b"), r"Section \1"),
    # "es see" / "S.C." / "S C" → "SC"
    (re.compile(r"\bes\s+see\b", re.IGNORECASE), "SC"),
    (re.compile(r"\bS\s*\.\s*C\s*\.\b"), "SC"),
    # "high court" / "High Court" — preserve capitalisation
    (re.compile(r"\bhigh court\b", re.IGNORECASE), "High Court"),
    (re.compile(r"\bsupreme court\b", re.IGNORECASE), "Supreme Court"),
    # "i p c" → "IPC"
    (re.compile(r"\bi\s+p\s+c\b", re.IGNORECASE), "IPC"),
    (re.compile(r"\bc\s+r\s+p\s+c\b", re.IGNORECASE), "CrPC"),
    (re.compile(r"\bc\s+p\s+c\b", re.IGNORECASE), "CPC"),
    # "N I Act" / "n i act" → "NI Act"
    (re.compile(r"\bn\s*[.,-]?\s*i\s*[.,-]?\s*act\b", re.IGNORECASE), "NI Act"),
    # "r e r a" → "RERA"
    (re.compile(r"\br\s+e\s+r\s+a\b", re.IGNORECASE), "RERA"),
    # Spacing in section references: "Section  138" → "Section 138"
    (re.compile(r"\b(Section|S\.?)\s{2,}(\d+)\b"), r"\1 \2"),
]


def _apply_legal_corrections(text: str) -> str:
    """Apply all legal term normalisation patterns to *text*."""
    for pattern, replacement in _LEGAL_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


# ─── Intent detection ─────────────────────────────────────────────────────────

_QUESTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(what|when|where|who|which|how|why|does|do|is|are|can|could|would|should)\b", re.IGNORECASE),
    re.compile(r"\?"),
]

_INSTRUCTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(draft|write|prepare|send|file|submit|create|generate|compose|issue|serve|attach)\b", re.IGNORECASE),
]


def detect_intent(text: str) -> LegalIntent:
    """
    Classify the utterance intent using keyword heuristics.

    Priority: question > instruction > narration > unknown.
    """
    if not text.strip():
        return "unknown"
    if any(p.search(text) for p in _QUESTION_PATTERNS):
        return "question"
    if any(p.search(text) for p in _INSTRUCTION_PATTERNS):
        return "instruction"
    # Non-empty text with no question/instruction signals → treat as narration
    return "narration"


# ─── Post-processor ───────────────────────────────────────────────────────────

class ASRPostprocessor:
    """
    Post-processes raw SegmentResult objects from the engine into
    clean ASRSegment objects ready for the ASRResult.

    Usage::

        pp = ASRPostprocessor()
        segments = pp.process_segments(raw_segments, engine_tag="whisper")
        transcript = pp.assemble_transcript(segments)
        intent = pp.detect_intent(transcript)
    """

    def __init__(
        self,
        no_speech_threshold: float = NO_SPEECH_THRESHOLD,
        enable_legal_corrections: bool = True,
    ) -> None:
        self._no_speech_threshold = no_speech_threshold
        self._enable_legal_corrections = enable_legal_corrections

    def process_segments(
        self,
        raw_segments: list[SegmentResult],
        engine_tag: str = "whisper",
    ) -> list[ASRSegment]:
        """
        Convert raw engine SegmentResults to clean ASRSegments.

        Steps per segment:
          1. Drop silence (no_speech_prob > threshold)
          2. Map avg_logprob → confidence
          3. Apply legal term normalisation
          4. Normalise whitespace
          5. Build ASRSegment (review_required auto-derived by Pydantic)
        """
        out: list[ASRSegment] = []
        for raw in raw_segments:
            # Filter silence
            if raw.no_speech_prob > self._no_speech_threshold:
                logger.debug(
                    "Dropping silent segment [%.1f–%.1f]: no_speech_prob=%.2f",
                    raw.start_s, raw.end_s, raw.no_speech_prob,
                )
                continue

            text = self._clean(raw.text)
            if not text:
                continue

            if self._enable_legal_corrections:
                text = _apply_legal_corrections(text)

            confidence = logprob_to_confidence(raw.avg_logprob)

            out.append(ASRSegment(
                segment_id=raw.segment_id,
                start_s=raw.start_s,
                end_s=raw.end_s,
                text=text,
                confidence=confidence,
                no_speech_prob=raw.no_speech_prob,
                review_required=False,   # Pydantic model_validator auto-sets
                asr_engine=engine_tag if engine_tag in {"whisper", "faster_whisper", "mock", "unknown"} else "unknown",  # type: ignore[arg-type]
                language=raw.language,
            ))
        return out

    def assemble_transcript(self, segments: list[ASRSegment]) -> str:
        """
        Concatenate segment texts in chronological order.

        Low-confidence segments are wrapped in [REVIEW]…[/REVIEW] markers
        so downstream modules can handle them separately.
        Mirrors the pattern from HTRPostprocessor.assemble_text().
        """
        parts: list[str] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if seg.review_required:
                parts.append(f"[REVIEW]{text}[/REVIEW]")
            else:
                parts.append(text)
        return " ".join(parts)

    def detect_intent(self, transcript: str) -> LegalIntent:
        """Classify transcript intent as question/instruction/narration/unknown."""
        return detect_intent(transcript)

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and strip control characters."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
