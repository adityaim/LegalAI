"""
modules/htr/postprocessor.py
──────────────────────────────
Post-processing for HTR output (Module 2).

Responsibilities:
  1. SymSpell spell-correction — fixes single-character HTR errors
     (e.g. "herahy" → "hereby", "Seotion" → "Section")
  2. Legal term preservation — skip correction for known legal terms
     to prevent over-correction of domain vocabulary
  3. Confidence flagging — mark regions with confidence < 70 for review
  4. Region type re-classification from geometry (fallback)
  5. Full-text assembly from all regions in reading order

SymSpell configuration
────────────────────────
SymSpell is a very fast edit-distance spell checker (O(1) lookup via
pre-computed delete neighbour dictionaries). We use the standard English
frequency dictionary bundled with symspellpy.

If symspellpy is not installed, the corrector silently no-ops — the raw
HTR text is returned unchanged with a debug log.

Legal term preservation
────────────────────────
A set of regex patterns matches legal identifiers that must NOT be
altered by spell correction:
  - Act / statute names: "NI Act", "IPC", "CrPC", "Contract Act"
  - Section references: "Section 138", "S. 302"
  - Citation patterns: "2024 SCC 123", "AIR 2021 SC"
  - Party names: detected as NP chunks or ALL-CAPS sequences
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from .schema import HTRRegion, HandwritingType

logger = logging.getLogger(__name__)

# ─── Confidence threshold ─────────────────────────────────────────────────────

LOW_CONFIDENCE_THRESHOLD = 70.0

# ─── Legal term patterns (must NOT be spell-corrected) ────────────────────────

_LEGAL_PRESERVE_PATTERNS: list[re.Pattern] = [
    # Statute abbreviations
    re.compile(r"\b(IPC|CrPC|CPC|NI Act|IT Act|RERA|NCLT|NCLAT|SC|HC|DC)\b"),
    # Section references
    re.compile(r"\bS(?:ec(?:tion)?)?\.?\s*\d+", re.IGNORECASE),
    # Case citation patterns: "2024 SCC 123" / "AIR 2021 SC 45"
    re.compile(r"\b\d{4}\s+(?:SCC|AIR|SCR|CLT|Cri LJ)\s+\d+\b"),
    # ALL-CAPS abbreviations (≥2 uppercase letters)
    re.compile(r"\b[A-Z]{2,}\b"),
    # Numeric strings
    re.compile(r"\b\d[\d\-/\.]*\b"),
]


def _has_legal_token(text: str) -> bool:
    """Return True if *text* contains any legal preservation pattern."""
    return any(p.search(text) for p in _LEGAL_PRESERVE_PATTERNS)


# ─── SymSpell corrector ───────────────────────────────────────────────────────

class SymSpellCorrector:
    """
    SymSpell-based spell corrector for HTR output.

    Lazy-loads the SymSpell model and dictionary on first call.
    Falls back to identity (no-op) if symspellpy is not installed.
    """

    _instance: "SymSpellCorrector | None" = None

    def __init__(self) -> None:
        self._sym_spell = None
        self._available: bool | None = None

    @classmethod
    def get_instance(cls) -> "SymSpellCorrector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def correct(self, text: str) -> str:
        """
        Correct spelling errors in *text*.

        Words matching legal preservation patterns are skipped.
        Returns *text* unchanged if SymSpell is not available.
        """
        if not self._load():
            return text
        if not text.strip():
            return text

        words = text.split()
        corrected: list[str] = []

        for word in words:
            # Preserve legal tokens, punctuation-only tokens, numbers
            if _has_legal_token(word) or not re.search(r"[a-zA-Z]", word):
                corrected.append(word)
                continue
            corrected.append(self._correct_word(word))

        return " ".join(corrected)

    def _correct_word(self, word: str) -> str:
        """Return best SymSpell correction for a single word."""
        try:
            suggestions = self._sym_spell.lookup(
                word.lower(),
                verbosity=0,          # best match only
                max_edit_distance=1,  # HTR errors are usually 1-char off
            )
            if suggestions:
                corrected = suggestions[0].term
                # Preserve original capitalisation
                if word[0].isupper():
                    corrected = corrected.capitalize()
                if word.isupper():
                    corrected = corrected.upper()
                return corrected
        except Exception:
            pass
        return word

    def _load(self) -> bool:
        """Lazy-load SymSpell + dictionary. Returns True if available."""
        if self._available is False:
            return False
        if self._sym_spell is not None:
            return True
        try:
            from symspellpy import SymSpell, Verbosity  # type: ignore[import]
            import importlib.resources as pkg_resources

            sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

            # Try bundled dictionary first
            try:
                import symspellpy
                dict_path = (
                    pkg_resources.files(symspellpy)
                    / "frequency_dictionary_en_82_765.txt"
                )
                loaded = sym_spell.load_dictionary(
                    str(dict_path), term_index=0, count_index=1
                )
            except Exception:
                loaded = False

            if not loaded:
                logger.warning(
                    "SymSpell dictionary not found — spell correction disabled. "
                    "Install with: pip install symspellpy"
                )
                self._available = False
                return False

            self._sym_spell = sym_spell
            self._available = True
            logger.debug("SymSpell corrector loaded")
            return True

        except ImportError:
            logger.info(
                "symspellpy not installed — spell correction skipped. "
                "Install with: pip install symspellpy"
            )
            self._available = False
            return False


# ─── Post-processor ───────────────────────────────────────────────────────────

class HTRPostprocessor:
    """
    Post-processes raw HTRRegion objects from the engine into clean output.

    Usage::

        pp = HTRPostprocessor()
        clean_regions = pp.process_regions(raw_regions)
        full_text = pp.assemble_text(clean_regions)
    """

    def __init__(self, enable_spellcheck: bool = True) -> None:
        self._corrector = SymSpellCorrector.get_instance() if enable_spellcheck else None

    def process_regions(self, regions: list[HTRRegion]) -> list[HTRRegion]:
        """
        Apply post-processing to each region:
        - Spell-correct text (with legal term preservation)
        - Set review_required based on confidence
        - Normalise whitespace
        """
        processed: list[HTRRegion] = []
        for region in regions:
            clean_text = self._clean(region.text)
            if self._corrector:
                clean_text = self._corrector.correct(clean_text)
            flagged = region.confidence < LOW_CONFIDENCE_THRESHOLD
            processed.append(region.model_copy(update={
                "text": clean_text,
                "review_required": flagged,
            }))
        return processed

    def assemble_text(self, regions: list[HTRRegion]) -> str:
        """
        Concatenate region texts in reading order.

        Low-confidence regions are wrapped in [REVIEW]…[/REVIEW] markers
        so downstream modules can handle them separately.
        """
        parts: list[str] = []
        for region in regions:
            text = region.text.strip()
            if not text:
                continue
            if region.review_required:
                parts.append(f"[REVIEW]{text}[/REVIEW]")
            else:
                parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _clean(text: str) -> str:
        """Basic text cleaning: strip control chars and normalise whitespace."""
        # Remove control chars (keep newlines)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Collapse multiple spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
