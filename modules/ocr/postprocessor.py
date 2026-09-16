"""
modules/ocr/postprocessor.py
──────────────────────────────
Legal-text post-processing for OCR output.

Responsibilities:
  1. **Legal text normalisation** — fix common OCR artefacts in legal text
     (section markers, act references, punctuation artefacts, ligatures)
  2. **Hyphenation repair** — join end-of-line hyphenated words
  3. **Whitespace normalisation** — collapse multiple spaces / blank lines
  4. **Low-confidence flagging** — mark blocks with confidence < threshold
  5. **Metadata extraction** — lightweight heuristic extraction of doc_type,
     parties, date, and jurisdiction from full text

All normalisation rules are designed to be conservative: they operate on
well-understood OCR artefact patterns and avoid modifying domain-specific
legal terminology.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

from .schema import BlockType, DocumentMetadata, OCRBlock

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

LOW_CONFIDENCE_THRESHOLD = 70.0   # blocks below this get review_required=True
PAGE_LOW_CONFIDENCE_THRESHOLD = 70.0  # pages where mean conf < this are flagged


# ─── Legal normalisation rules ───────────────────────────────────────────────

# (pattern, replacement) — applied in order
_NORMALISATION_RULES: list[tuple[re.Pattern, str]] = [
    # Fix section abbreviations:  "S. 138" / "Sec.138" → "Section 138"
    (re.compile(r"\bS\.\s*(\d)", re.IGNORECASE), r"Section \1"),
    (re.compile(r"\bSec\.\s*(\d)", re.IGNORECASE), r"Section \1"),

    # Fix clause abbreviations: "cl. 4" → "Clause 4"
    (re.compile(r"\bcl\.\s*(\d)", re.IGNORECASE), r"Clause \1"),

    # Fix article abbreviations: "Art. 14" → "Article 14"
    (re.compile(r"\bArt\.\s*(\d)", re.IGNORECASE), r"Article \1"),

    # Remove stray pipe characters (common OCR artefact for 'l' / 'I')
    # Only remove pipes that are surrounded by whitespace / line boundaries
    (re.compile(r"(?<!\S)\|(?!\S)"), " "),

    # Fix broken ligatures: common OCR confusion patterns
    (re.compile(r"\bfl\b"), "fi"),   # fl ligature → fi when isolated
    (re.compile(r"\bft\b"), "ft"),   # no-op but shows pattern structure

    # Normalise curly quotes to straight quotes
    (re.compile(r"[\u2018\u2019]"), "'"),
    (re.compile(r"[\u201c\u201d]"), '"'),

    # Remove null characters and other control chars (except newline/tab)
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"), ""),

    # Collapse 3+ consecutive newlines to 2
    (re.compile(r"\n{3,}"), "\n\n"),
]

# End-of-line hyphenation: "inter-\npretation" → "interpretation"
_HYPHEN_RE = re.compile(r"(\w+)-\n(\w+)")

# ─── Document type detection ─────────────────────────────────────────────────

_DOC_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bFIRST INFORMATION REPORT\b|\bFIR\b", re.IGNORECASE), "fir"),
    (re.compile(r"\bCONTRACT\b|\bAGREEMENT\b", re.IGNORECASE), "contract"),
    (re.compile(r"\bAFFIDAVIT\b", re.IGNORECASE), "affidavit"),
    (re.compile(r"\bCOURT ORDER\b|\bORDER\b.*\bCOURT\b", re.IGNORECASE), "court_order"),
    (re.compile(r"\bJUDGMENT\b|\bJUDGEMENT\b", re.IGNORECASE), "judgment"),
    (re.compile(r"\bNOTICE\b", re.IGNORECASE), "notice"),
    (re.compile(r"\bDEED\b", re.IGNORECASE), "deed"),
    (re.compile(r"\bWRIT PETITION\b|\bWRIT\b", re.IGNORECASE), "writ_petition"),
]

# ─── Party extraction ────────────────────────────────────────────────────────

# Common legal party indicators in Indian legal documents
_PARTY_RE = re.compile(
    r"(?:BETWEEN|PARTY(?:\s+OF)?|PLAINTIFF|DEFENDANT|PETITIONER|RESPONDENT"
    r"|APPELLANT|versus|vs\.?|V/S)\s*[:\-]?\s*"
    r"([A-Z][A-Za-z\s\.&,]{3,50}(?:Ltd|LLP|Pvt|Inc|Corp|Pvt\. Ltd\.?)?)",
    re.IGNORECASE,
)

# ─── Date extraction ─────────────────────────────────────────────────────────

_DATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\b"),
    re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
               r"September|October|November|December)\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(January|February|March|April|May|June|July|August|"
               r"September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
               re.IGNORECASE),
]

_MONTH_MAP = {m.lower(): i + 1 for i, m in enumerate(
    ["january","february","march","april","may","june",
     "july","august","september","october","november","december"]
)}

# ─── Jurisdiction extraction ─────────────────────────────────────────────────

_INDIAN_JURISDICTIONS = [
    "Delhi", "Mumbai", "Bombay", "Kolkata", "Calcutta", "Chennai", "Madras",
    "Bangalore", "Bengaluru", "Hyderabad", "Ahmedabad", "Pune", "Jaipur",
    "Lucknow", "Allahabad", "Patna", "Guwahati", "Chandigarh",
    "Supreme Court", "High Court", "District Court", "Sessions Court",
    "Magistrate Court", "National Consumer", "NCLAT", "NCLT", "RERA",
    "Andhra Pradesh", "Bihar", "Gujarat", "Haryana", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Punjab", "Rajasthan",
    "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal",
]
_JURISDICTION_RE = re.compile(
    r"(" + "|".join(re.escape(j) for j in _INDIAN_JURISDICTIONS) + r")",
    re.IGNORECASE,
)


# ─── Post-processor class ────────────────────────────────────────────────────

class LegalTextPostprocessor:
    """
    Post-processes raw OCR blocks into clean, normalised legal text.

    Usage::

        pp = LegalTextPostprocessor()
        clean_blocks = pp.process_blocks(raw_blocks)
        metadata = pp.extract_metadata(full_text)
        low_conf_pages = pp.low_confidence_pages(clean_blocks)
    """

    def process_blocks(self, blocks: list[OCRBlock]) -> list[OCRBlock]:
        """
        Apply normalisation and confidence flagging to each block.

        Returns a new list of OCRBlock objects with cleaned text and
        ``review_required`` flags set.
        """
        processed: list[OCRBlock] = []
        for block in blocks:
            clean_text = self._normalise(block.text)
            flagged = block.confidence < LOW_CONFIDENCE_THRESHOLD
            processed.append(block.model_copy(update={
                "text": clean_text,
                "review_required": flagged,
            }))
        return processed

    def merge_blocks_to_text(self, blocks: list[OCRBlock]) -> str:
        """
        Produce the ``full_text`` string from a list of processed blocks.

        Headings are separated by double newlines; body blocks by single
        newlines; footers and page numbers are omitted from the main text.
        """
        parts: list[str] = []
        for block in blocks:
            if block.type in ("footer", "page_number"):
                continue
            text = block.text.strip()
            if not text:
                continue
            if block.type == "heading":
                parts.append(f"\n\n{text}\n")
            else:
                parts.append(text)
        return "\n".join(parts).strip()

    def extract_metadata(self, full_text: str) -> DocumentMetadata:
        """
        Extract lightweight document metadata from *full_text* using regex patterns.

        This is a best-effort extraction — results may be partial.
        Downstream modules (NER, LLM) are expected to refine these.
        """
        doc_type = self._detect_doc_type(full_text)
        parties = self._extract_parties(full_text)
        date = self._extract_date(full_text)
        jurisdiction = self._extract_jurisdiction(full_text)
        return DocumentMetadata(
            doc_type=doc_type,
            parties=parties,
            date=date,
            jurisdiction=jurisdiction,
        )

    def low_confidence_pages(self, blocks: list[OCRBlock]) -> list[int]:
        """
        Return 1-indexed page numbers where the mean block confidence is below
        the page-level threshold.
        """
        from collections import defaultdict
        page_confs: dict[int, list[float]] = defaultdict(list)
        for block in blocks:
            page_confs[block.page].append(block.confidence)

        flagged: list[int] = []
        for page_num, confs in sorted(page_confs.items()):
            if confs:
                mean_conf = sum(confs) / len(confs)
                if mean_conf < PAGE_LOW_CONFIDENCE_THRESHOLD:
                    flagged.append(page_num)
        return flagged

    # ── Normalisation helpers ──────────────────────────────────────────────

    def _normalise(self, text: str) -> str:
        """Apply all normalisation rules to *text*."""
        # Repair end-of-line hyphens first (before collapsing newlines)
        text = _HYPHEN_RE.sub(r"\1\2", text)

        for pattern, replacement in _NORMALISATION_RULES:
            text = pattern.sub(replacement, text)

        # Final whitespace normalisation
        text = re.sub(r"[ \t]+", " ", text)  # collapse multiple spaces
        text = text.strip()
        return text

    # ── Metadata helpers ───────────────────────────────────────────────────

    @staticmethod
    def _detect_doc_type(text: str) -> str | None:
        for pattern, doc_type in _DOC_TYPE_PATTERNS:
            if pattern.search(text):
                return doc_type
        return None

    @staticmethod
    def _extract_parties(text: str) -> list[str]:
        matches = _PARTY_RE.findall(text)
        # Deduplicate while preserving order
        seen: set[str] = set()
        parties: list[str] = []
        for m in matches:
            cleaned = m.strip().rstrip(",")
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                parties.append(cleaned)
        return parties[:6]  # cap at 6 parties to avoid false positives

    @staticmethod
    def _extract_date(text: str) -> str | None:
        """Return the first parseable date found in *text* as ISO 8601 string."""
        for pattern in _DATE_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            groups = m.groups()
            try:
                if len(groups) == 3:
                    # Check if groups[1] is a month name
                    if groups[1].lower() in _MONTH_MAP:
                        day, month, year = int(groups[0]), _MONTH_MAP[groups[1].lower()], int(groups[2])
                    elif groups[0].lower() in _MONTH_MAP:
                        month, day, year = _MONTH_MAP[groups[0].lower()], int(groups[1]), int(groups[2])
                    else:
                        day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    dt = datetime(year, month, day)
                    return dt.strftime("%Y-%m-%d")
            except (ValueError, KeyError):
                continue
        return None

    @staticmethod
    def _extract_jurisdiction(text: str) -> str | None:
        m = _JURISDICTION_RE.search(text)
        return m.group(1) if m else None


# ─── Module-level convenience instance ───────────────────────────────────────

_default_postprocessor = LegalTextPostprocessor()


def postprocess_blocks(blocks: list[OCRBlock]) -> list[OCRBlock]:
    """Convenience wrapper around ``LegalTextPostprocessor.process_blocks``."""
    return _default_postprocessor.process_blocks(blocks)


def extract_metadata(full_text: str) -> DocumentMetadata:
    """Convenience wrapper around ``LegalTextPostprocessor.extract_metadata``."""
    return _default_postprocessor.extract_metadata(full_text)
