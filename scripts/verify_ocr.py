#!/usr/bin/env python3
"""
scripts/verify_ocr.py
──────────────────────
Quick smoke-test / demo script for the OCR module.

Usage::

    python scripts/verify_ocr.py --input path/to/document.pdf
    python scripts/verify_ocr.py --input scan.png --output result.json
    python scripts/verify_ocr.py --input contract.pdf --engine tesseract --lang hi
    python scripts/verify_ocr.py --demo   # generate and process a synthetic PDF

Prints a human-readable summary and validates the output against the schema.
Optionally writes the full JSON result to --output.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.ocr import OCRDocumentReader, OCRResult
from modules.ocr.reader import ReaderConfig


# ─── ANSI colours ────────────────────────────────────────────────────────────

def green(s: str) -> str:  return f"\033[92m{s}\033[0m"
def yellow(s: str) -> str: return f"\033[93m{s}\033[0m"
def red(s: str) -> str:    return f"\033[91m{s}\033[0m"
def bold(s: str) -> str:   return f"\033[1m{s}\033[0m"

OK = "[OK] "
WARN = "[!]  "
FAIL = "[FAIL]"


# ─── Demo PDF generator ───────────────────────────────────────────────────────

def generate_demo_pdf() -> Path:
    """Create a small synthetic legal PDF for demo purposes."""
    try:
        import fitz
    except ImportError:
        print(red("PyMuPDF not installed — cannot generate demo PDF."))
        print("Install with: pip install pymupdf")
        sys.exit(1)

    import io
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    sample_text = [
        ("THIS AGREEMENT", 72, 80, 20),
        ("This Agreement is made between Alpha Technologies Pvt. Ltd., a company", 72, 120, 12),
        ("incorporated under the Companies Act, 2013, having its registered office", 72, 138, 12),
        ("at 123 Business Park, Delhi - 110001 ('Party A'), and Beta Legal Services", 72, 156, 12),
        ("LLP, registered at 456 Law Avenue, Mumbai - 400001 ('Party B').", 72, 174, 12),
        ("1. Definitions", 72, 210, 14),
        ("1.1 'Agreement' means this contract and all schedules attached hereto.", 72, 232, 12),
        ("1.2 'Services' means the legal advisory services described in Schedule A.", 72, 250, 12),
        ("2. Term and Termination", 72, 290, 14),
        ("This Agreement shall commence on 15 January 2025 and continue for a period", 72, 312, 12),
        ("of one (1) year, unless terminated earlier in accordance with Section 8.", 72, 330, 12),
        ("IN WITNESS WHEREOF the parties have executed this Agreement as of the date", 72, 700, 11),
        ("first written above.", 72, 716, 11),
        ("Page 1", 280, 800, 9),
    ]

    for text, x, y, size in sample_text:
        page.insert_text((x, y), text, fontsize=size)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(buf.getvalue())
    tmp.close()
    return Path(tmp.name)


# ─── Summary printer ──────────────────────────────────────────────────────────

def print_summary(result: OCRResult) -> None:
    print()
    print(bold("=" * 60))
    print(bold("  OCR Module — Processing Result"))
    print(bold("=" * 60))
    print(f"  Document ID   : {result.document_id}")
    print(f"  Pages         : {result.pages}")
    print(f"  Language      : {result.language}")
    print(f"  Blocks        : {len(result.blocks)}")
    print(f"  Processing    : {result.processing_time_s:.2f}s")
    print(f"  Mean Confidence: {result.mean_confidence():.1f}%")

    if result.low_confidence_pages:
        print(f"  " + yellow(f"{WARN} Low-confidence pages: {result.low_confidence_pages}"))
    else:
        print(f"  " + green(f"{OK} All pages above confidence threshold"))

    review_pages = result.pages_needing_review()
    if review_pages:
        print(f"  " + yellow(f"{WARN} Pages needing human review: {review_pages}"))

    # Metadata
    m = result.metadata
    print()
    print(bold("  Document Metadata"))
    print(f"  Doc Type      : {m.doc_type or '(not detected)'}")
    print(f"  Parties       : {', '.join(m.parties) if m.parties else '(none detected)'}")
    print(f"  Date          : {m.date or '(not detected)'}")
    print(f"  Jurisdiction  : {m.jurisdiction or '(not detected)'}")

    # Block breakdown
    print()
    print(bold("  Blocks by Type"))
    from collections import Counter
    counts = Counter(b.type for b in result.blocks)
    for btype, count in sorted(counts.items()):
        print(f"    {btype:<15}: {count}")

    # Text preview
    print()
    print(bold("  Text Preview (first 400 chars)"))
    print("-" * 60)
    preview = result.full_text[:400].replace("\n", "\n  ")
    print(f"  {preview}")
    if len(result.full_text) > 400:
        print(f"  ... [{len(result.full_text)} chars total]")
    print("-" * 60)


# ─── Schema validation ────────────────────────────────────────────────────────

def validate_schema(result: OCRResult) -> bool:
    """Validate result by round-tripping through model_dump → model_validate."""
    from pydantic import ValidationError
    try:
        data = result.model_dump()
        OCRResult.model_validate(data)
        print(green(f"{OK} Schema validation passed"))
        return True
    except ValidationError as e:
        print(red(f"{FAIL} Schema validation FAILED: {e}"))
        return False


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the OCR module by processing a PDF or image file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", help="Path to PDF or image file")
    parser.add_argument("--output", "-o", help="Write full JSON result to this file")
    parser.add_argument(
        "--engine", default="paddle",
        choices=["paddle", "tesseract"],
        help="OCR engine to use (default: paddle)",
    )
    parser.add_argument(
        "--layout", default="heuristic",
        choices=["heuristic", "layoutparser"],
        help="Layout detection engine (default: heuristic)",
    )
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a synthetic demo PDF and process it")
    args = parser.parse_args()

    # ── Determine input path ───────────────────────────────────────────────
    if args.demo:
        print(yellow("Generating synthetic demo PDF…"))
        input_path = generate_demo_pdf()
        print(f"  Demo PDF: {input_path}")
        demo_cleanup = True
    elif args.input:
        input_path = Path(args.input)
        demo_cleanup = False
    else:
        parser.print_help()
        sys.exit(1)

    # ── Configure reader ───────────────────────────────────────────────────
    config = ReaderConfig(
        language=args.lang,
        ocr_engine=args.engine,        # type: ignore[arg-type]
        layout_engine=args.layout,     # type: ignore[arg-type]
    )
    reader = OCRDocumentReader(config=config)

    # ── Process ────────────────────────────────────────────────────────────
    print(f"\nProcessing: {bold(str(input_path))}")
    try:
        result = reader.process(input_path)
    except Exception as exc:
        print(red(f"\n{FAIL}  Processing failed: {exc}"))
        sys.exit(1)
    finally:
        if args.demo:
            input_path.unlink(missing_ok=True)

    # ── Print summary ──────────────────────────────────────────────────────
    print_summary(result)

    # ── Validate schema ────────────────────────────────────────────────────
    print()
    valid = validate_schema(result)

    # ── Write output ───────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2,
                      default=str)
        print(green(f"{OK} Full result written to: {out_path}"))

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
