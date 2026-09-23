"""
modules/fusion/cleaner.py
──────────────────────────
Text cleaning and normalisation for Module 4 (Text Fusion).

Responsibilities:
  1. Strip control characters (keep newlines, tabs normalised to spaces)
  2. Normalise Unicode — NFKC decomposition (fixes ligatures, half-width chars)
  3. Collapse excessive whitespace / blank lines
  4. Normalise Indian legal punctuation and common Whisper mis-spacings
  5. Near-duplicate sentence deduplication — remove sentences that appear
     in more than one modality (e.g. the same header OCR'd AND whispered)
  6. Extract [REVIEW]…[/REVIEW] spans into ReviewFlag objects

Design notes
─────────────
• Deduplication is token-set similarity (Jaccard ≥ 0.85 on word sets) not
  exact match — handles minor OCR/ASR spelling variation of the same sentence.
• Dedup is CROSS-modality only: within a single modality block sentences
  are kept as-is (they may carry important repetition like numbered clauses).
• [REVIEW] markers are preserved through cleaning so downstream modules
  (RAG, LLM) can weight low-confidence spans appropriately.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from .schema import ModalitySource, ReviewFlag


# ─── Constants ────────────────────────────────────────────────────────────────

# Jaccard similarity threshold for cross-modality dedup
_DEDUP_THRESHOLD = 0.85

# Regex to extract [REVIEW]…[/REVIEW] spans
_REVIEW_RE = re.compile(r"\[REVIEW\](.*?)\[/REVIEW\]", re.DOTALL)

# Patterns for legal punctuation normalisation
_LEGAL_NORM: list[tuple[re.Pattern, str]] = [
    # Multiple spaces between legal terms
    (re.compile(r"(?<=\w)\.(?=\w)"), ". "),          # "Section.138" → "Section. 138"
    # Comma after section: "Section138," → "Section 138,"
    (re.compile(r"\b(Section|S\.?|Sec\.?)\s*(\d+)\b", re.IGNORECASE), r"Section \2"),
    # Smart quotes → ASCII
    (re.compile(r"[\u2018\u2019]"), "'"),
    (re.compile(r"[\u201c\u201d]"), '"'),
    # Em/en dashes → hyphen-space
    (re.compile(r"[\u2013\u2014]"), " - "),
    # Ellipsis
    (re.compile(r"\u2026"), "..."),
    # Non-breaking spaces
    (re.compile(r"\u00a0"), " "),
]


# ─── Core text cleaner ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply full normalisation pipeline to a single modality text block.

    Steps:
      1. NFKC Unicode normalisation
      2. Remove control characters (keep \\n \\t)
      3. Legal punctuation normalisation
      4. Collapse whitespace (multiple spaces → one; 3+ newlines → 2)
      5. Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # 1. NFKC normalise (fixes ligatures, half-width chars, decomposed accents)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove control chars (keep newline \x0a and tab \x09)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Legal punctuation normalisation
    for pattern, replacement in _LEGAL_NORM:
        text = pattern.sub(replacement, text)

    # 4. Whitespace normalisation
    text = re.sub(r"[ \t]+", " ", text)         # multiple spaces → one
    text = re.sub(r" +\n", "\n", text)           # trailing spaces before newline
    text = re.sub(r"\n{3,}", "\n\n", text)       # 3+ newlines → 2

    return text.strip()


# ─── [REVIEW] span extractor ─────────────────────────────────────────────────

def extract_review_flags(
    text: str,
    modality: ModalitySource,
) -> list[ReviewFlag]:
    """
    Find all [REVIEW]…[/REVIEW] spans in *text* and return ReviewFlag objects.

    The spans remain in the text as-is (not removed here) so downstream
    modules can still see the markers.
    """
    flags: list[ReviewFlag] = []
    for match in _REVIEW_RE.finditer(text):
        content = match.group(1).strip()
        if content:
            flags.append(ReviewFlag(
                modality=modality,
                text=content,
                reason="low_confidence",
            ))
    return flags


# ─── Cross-modality deduplicator ──────────────────────────────────────────────

def _sentence_tokens(sentence: str) -> frozenset[str]:
    """Lowercase word tokens from a sentence (ignore [REVIEW] markers)."""
    clean = _REVIEW_RE.sub(" ", sentence)
    return frozenset(re.findall(r"\b\w+\b", clean.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def deduplicate_cross_modality(blocks: list[tuple[str, str]]) -> list[str]:
    """
    Remove near-duplicate sentences that appear across multiple modalities.

    Parameters
    ----------
    blocks : list of (source_tag, text)
        Each tuple is one modality block. The FIRST occurrence of a sentence
        is kept; subsequent near-duplicates in later modalities are dropped.

    Returns
    -------
    list[str]
        Deduplicated texts in the same order as *blocks*.
    """
    seen_tokens: list[frozenset[str]] = []   # sentence token sets already emitted
    result: list[str] = []

    for _source, text in blocks:
        sentences = _split_sentences(text)
        kept: list[str] = []
        for sent in sentences:
            tok = _sentence_tokens(sent)
            if not tok:
                kept.append(sent)
                continue
            # Check against all previously emitted sentences
            is_dup = any(
                _jaccard(tok, seen) >= _DEDUP_THRESHOLD
                for seen in seen_tokens
            )
            if not is_dup:
                kept.append(sent)
                seen_tokens.append(tok)
            # If duplicate: silently drop the sentence
        result.append(" ".join(kept))

    return result
