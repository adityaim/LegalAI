#!/usr/bin/env python3
"""
scripts/verify_htr.py
──────────────────────
End-to-end smoke test and demo for Module 2: Handwritten Document Reader.

Usage::

    # Quick demo — generates a synthetic handwriting image and processes it
    python scripts/verify_htr.py --demo

    # Process a real image or PDF page
    python scripts/verify_htr.py --input path/to/annotated_contract.jpg

    # Save full JSON output
    python scripts/verify_htr.py --input scan.png --output htr_result.json

    # Hindi/Devanagari handwriting
    python scripts/verify_htr.py --input hindi_note.jpg --lang hi

    # Check what engines are available
    python scripts/verify_htr.py --health
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.htr import HandwritingReader, HTRResult
from modules.htr.reader import ReaderConfig

# ─── ANSI / ASCII output helpers ─────────────────────────────────────────────

def green(s: str) -> str:  return f"\033[92m{s}\033[0m"
def yellow(s: str) -> str: return f"\033[93m{s}\033[0m"
def red(s: str) -> str:    return f"\033[91m{s}\033[0m"
def bold(s: str) -> str:   return f"\033[1m{s}\033[0m"

OK   = "[OK]  "
WARN = "[!]   "
FAIL = "[FAIL]"


# ─── Demo image generator ─────────────────────────────────────────────────────

def generate_demo_image() -> Path:
    """
    Create a synthetic document image with printed text and handwritten annotations.
    Returns the path to a temporary PNG file.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print(red(f"{FAIL} OpenCV not installed. Run: pip install opencv-python-headless"))
        sys.exit(1)

    h, w = 700, 600
    img = np.full((h, w, 3), 252, dtype=np.uint8)  # off-white background

    # ── Printed text (simulated with OpenCV) ──────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "THIS AGREEMENT", (50, 70), font, 1.0, (30, 30, 30), 2)
    cv2.putText(img, "This Agreement is made between Alpha Ltd and Beta Corp",
                (50, 120), font, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "on the terms and conditions set forth herein.",
                (50, 145), font, 0.5, (50, 50, 50), 1)

    # ── Horizontal rules (printed table lines) ────────────────────────────
    cv2.line(img, (50, 200), (550, 200), (180, 180, 180), 1)
    cv2.line(img, (50, 250), (550, 250), (180, 180, 180), 1)
    cv2.putText(img, "Party A:", (60, 235), font, 0.45, (80, 80, 80), 1)
    cv2.putText(img, "Party B:", (60, 285), font, 0.45, (80, 80, 80), 1)
    cv2.line(img, (50, 300), (550, 300), (180, 180, 180), 1)

    # ── Handwritten fill-ins (simulated with thick polylines) ─────────────
    # Party A fill-in
    pts_a = np.array([[180, 225], [220, 215], [270, 228], [330, 218], [380, 226]], np.int32)
    cv2.polylines(img, [pts_a], False, (10, 10, 60), 3)

    # Party B fill-in
    pts_b = np.array([[180, 275], [240, 265], [300, 278], [360, 268]], np.int32)
    cv2.polylines(img, [pts_b], False, (10, 10, 60), 3)

    # ── Margin annotation ─────────────────────────────────────────────────
    ann_pts = [
        np.array([[520, 120], [540, 112], [555, 125], [565, 115]], np.int32),
        np.array([[515, 138], [542, 130], [560, 142]], np.int32),
    ]
    for pts in ann_pts:
        cv2.polylines(img, [pts], False, (30, 0, 120), 2)
    # Arrow pointing to text
    cv2.arrowedLine(img, (510, 128), (460, 135), (80, 0, 160), 1, tipLength=0.3)

    # ── Signature block ───────────────────────────────────────────────────
    cv2.putText(img, "Signature:", (50, 580), font, 0.45, (80, 80, 80), 1)
    cv2.line(img, (160, 580), (400, 580), (200, 200, 200), 1)
    sig_pts = np.array([
        [165, 570], [195, 555], [230, 568], [265, 553],
        [295, 562], [330, 551], [360, 565], [390, 558]
    ], np.int32)
    cv2.polylines(img, [sig_pts], False, (20, 20, 20), 2)

    # ── Standalone handwritten note ───────────────────────────────────────
    note_lines = [
        np.array([[60, 400], [100, 392], [150, 405], [200, 395], [250, 408]], np.int32),
        np.array([[60, 422], [110, 415], [170, 428], [230, 418], [280, 430]], np.int32),
        np.array([[60, 444], [120, 437], [180, 449], [220, 440]], np.int32),
    ]
    for line in note_lines:
        cv2.polylines(img, [line], False, (0, 50, 20), 3)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, img)
    tmp.close()
    return Path(tmp.name)


# ─── Health check ─────────────────────────────────────────────────────────────

def check_health() -> None:
    print(bold("\n  Module 2 (HTR) — Engine Availability"))
    print("=" * 52)

    checks = [
        ("OpenCV (segmenter)",   "cv2",            True),
        ("transformers (TrOCR)", "transformers",   False),
        ("torch",                "torch",          False),
        ("pytesseract (Hindi)",  "pytesseract",    False),
        ("symspellpy",           "symspellpy",     False),
        ("PyMuPDF (PDF input)",  "fitz",           False),
    ]

    all_required_ok = True
    for label, module, required in checks:
        try:
            __import__(module)
            print(green(f"  {OK}{label}"))
        except ImportError:
            status = FAIL if required else WARN
            note = "REQUIRED" if required else "optional"
            print((red if required else yellow)(f"  {status}{label}  [{note}]"))
            if required:
                all_required_ok = False

    print()
    if all_required_ok:
        print(green(f"  {OK}All required dependencies present"))
    else:
        print(red(f"  {FAIL}Missing required dependencies — see above"))
    print()


# ─── Result printer ───────────────────────────────────────────────────────────

def print_summary(result: HTRResult, input_path: str) -> None:
    print()
    print(bold("=" * 60))
    print(bold("  Module 2 (HTR) — Processing Result"))
    print(bold("=" * 60))
    print(f"  Document ID    : {result.document_id}")
    print(f"  Input          : {Path(input_path).name}")
    print(f"  Input type     : {result.input_type}")
    print(f"  Language       : {result.language}")
    print(f"  Regions found  : {len(result.regions)}")
    print(f"  Processing     : {result.processing_time_s:.2f}s")

    if result.regions:
        print(f"  Mean confidence: {result.mean_confidence():.1f}%")
    else:
        print(f"  Mean confidence: N/A (no handwriting detected)")

    if result.low_confidence_regions:
        print(yellow(f"  {WARN}{result.low_confidence_regions} region(s) need human review"))
    else:
        print(green(f"  {OK}All regions above confidence threshold"))

    # Region breakdown by type
    if result.regions:
        print()
        print(bold("  Regions by Type"))
        from collections import Counter
        counts = Counter(r.type for r in result.regions)
        for rtype, count in sorted(counts.items()):
            print(f"    {rtype:<20}: {count}")

        # Individual regions
        print()
        print(bold("  Detected Regions"))
        print(f"  {'#':<4} {'Type':<18} {'Conf':>6}  {'Review':>7}  Text")
        print("  " + "-" * 56)
        for i, region in enumerate(result.regions, 1):
            flag = "YES" if region.review_required else "no"
            text_preview = (region.text or "(empty)")[:30].replace("\n", " ")
            conf_str = f"{region.confidence:.0f}%"
            print(f"  {i:<4} {region.type:<18} {conf_str:>6}  {flag:>7}  {text_preview}")

    # Full text preview
    print()
    print(bold("  Assembled Text (first 400 chars)"))
    print("  " + "-" * 56)
    preview = (result.full_text or "(no text extracted)")[:400].replace("\n", "\n  ")
    print(f"  {preview}")
    if len(result.full_text) > 400:
        print(f"  ... [{len(result.full_text)} chars total]")
    print("  " + "-" * 56)

    # Fusion text preview (with [REVIEW] markers)
    fusion = result.to_fusion_text()
    if fusion and fusion != result.full_text:
        print()
        print(bold("  Fusion Text (for Module 4, with [REVIEW] markers)"))
        print("  " + "-" * 56)
        print(f"  {fusion[:300].replace(chr(10), chr(10)+'  ')}")
        print("  " + "-" * 56)


# ─── Schema validation ────────────────────────────────────────────────────────

def validate_schema(result: HTRResult) -> bool:
    from pydantic import ValidationError
    try:
        HTRResult.model_validate(result.model_dump())
        print(green(f"  {OK}Schema validation passed"))
        return True
    except ValidationError as exc:
        print(red(f"  {FAIL}Schema validation FAILED:\n{exc}"))
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Module 2 (HTR) by processing a document image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",   "-i", help="Path to image or PDF file")
    parser.add_argument("--output",  "-o", help="Write full JSON result to this file")
    parser.add_argument("--lang",    default="en", help="Language code (default: en)")
    parser.add_argument("--segmenter", default="heuristic",
                        choices=["heuristic", "yolov8"],
                        help="Segmentation backend (default: heuristic)")
    parser.add_argument("--trocr-variant", default="base",
                        choices=["base", "large", "small"],
                        dest="trocr_variant",
                        help="TrOCR model size (default: base)")
    parser.add_argument("--no-spellcheck", action="store_true",
                        help="Disable SymSpell post-correction")
    parser.add_argument("--demo",   action="store_true",
                        help="Generate a synthetic demo image and process it")
    parser.add_argument("--health", action="store_true",
                        help="Check engine availability and exit")
    args = parser.parse_args()

    if args.health:
        check_health()
        return

    # ── Resolve input ──────────────────────────────────────────────────────
    demo_cleanup = False
    if args.demo:
        print(yellow("Generating synthetic demo image with handwritten annotations..."))
        input_path = generate_demo_image()
        print(f"  Demo image: {input_path}")
        demo_cleanup = True
    elif args.input:
        input_path = Path(args.input)
    else:
        parser.print_help()
        sys.exit(1)

    # ── Configure reader ───────────────────────────────────────────────────
    config = ReaderConfig(
        language=args.lang,
        segmenter_engine=args.segmenter,           # type: ignore[arg-type]
        trocr_variant=args.trocr_variant,          # type: ignore[arg-type]
        enable_spellcheck=not args.no_spellcheck,
    )
    reader = HandwritingReader(config=config)

    # ── Process ────────────────────────────────────────────────────────────
    print(f"\nProcessing: {bold(str(input_path))}")
    try:
        result = reader.process(input_path)
    except Exception as exc:
        print(red(f"\n  {FAIL}Processing failed: {exc}"))
        sys.exit(1)
    finally:
        if demo_cleanup:
            input_path.unlink(missing_ok=True)

    # ── Print summary ──────────────────────────────────────────────────────
    print_summary(result, str(input_path))

    # ── Schema validation ──────────────────────────────────────────────────
    print()
    valid = validate_schema(result)

    # ── Write output ───────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        print(green(f"  {OK}Full result written to: {out_path}"))

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
