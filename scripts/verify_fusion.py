"""
scripts/verify_fusion.py
─────────────────────────
CLI verification and demo tool for Module 4: Text Fusion & Pre-processing.

Usage
─────
  # Synthetic demo (mock OCR/HTR/ASR results, shows full pipeline output)
  python -m scripts.verify_fusion --demo

  # Typed-text only mode (no other modalities required)
  python -m scripts.verify_fusion --typed "What does Section 138 NI Act say?"

  # REST API (server must be running)
  curl -X POST http://localhost:8001/api/v1/fusion/fuse \\
       -H "Content-Type: application/json" \\
       -d '{"typed_text": "Draft a legal notice under Section 138."}'
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# ─── Demo ─────────────────────────────────────────────────────────────────────

def _run_demo() -> None:
    print("\n=== Module 4: Text Fusion & Pre-processing — Demo ===\n")

    from modules.fusion import TextFusionProcessor
    from modules.fusion.schema import FusedDocument

    # ── Synthetic inputs (matching what M1/M2/M3 would produce) ─────────
    class _FakeOCR:
        document_id = "ocr-uuid-001"
        full_text = (
            "THIS AGREEMENT is entered into as of 15 January 2025 between "
            "Alpha Technologies Pvt. Ltd. (hereinafter 'Alpha') and "
            "Beta Legal Services LLP (hereinafter 'Beta').\n\n"
            "Clause 1: Alpha agrees to pay Beta within 30 days of invoice date.\n"
            "Clause 2: Disputes shall be settled under the Arbitration Act 1996."
        )
        language = "en"

    class _FakeHTR:
        document_id = "htr-uuid-002"
        language = "en"
        def to_fusion_text(self) -> str:
            return (
                "Signed by R. K. Sharma\n"
                "[REVIEW]see clause 4 re: penalty[/REVIEW]\n"
                "Date: 15-01-2025"
            )

    class _FakeASR:
        document_id = "asr-uuid-003"
        language = "en"
        intent = "question"
        def to_fusion_text(self) -> str:
            return (
                "What are the payment terms under Clause 1? "
                "Does Section 138 of the NI Act apply if Alpha defaults?"
            )

    fuser = TextFusionProcessor()
    t0 = time.perf_counter()
    doc: FusedDocument = fuser.fuse(
        ocr=_FakeOCR(),
        htr=_FakeHTR(),
        asr=_FakeASR(),
        typed_text="Summarise the key obligations of Alpha Technologies.",
    )
    elapsed = time.perf_counter() - t0

    # ── Display ─────────────────────────────────────────────────────────
    print(f"  Modalities fused : {len(doc.modalities)}"
          f"  ({', '.join(b.source for b in doc.modalities)})")
    print(f"  Primary language : {doc.primary_language}")
    print(f"  Detected intent  : {doc.detected_intent}")
    print(f"  Total words      : {doc.total_word_count}")
    print(f"  Review flags     : {len(doc.review_flags)}")
    print(f"  Processing       : {elapsed*1000:.1f} ms")
    print()

    print("  ── fused_text ──────────────────────────────────────────────────")
    for line in doc.fused_text.splitlines():
        print("  " + line)
    print()

    print("  ── for_rag() (tag-stripped) ────────────────────────────────────")
    for line in doc.for_rag().splitlines()[:8]:
        print("  " + line)
    print()

    print("  ── for_llm() (with header) ─────────────────────────────────────")
    for line in doc.for_llm().splitlines()[:10]:
        print("  " + line)
    print("  ...")
    print()

    if doc.review_flags:
        print("  ── Review flags ────────────────────────────────────────────────")
        for f in doc.review_flags:
            print(f"  [{f.modality}] {f.text}")
        print()

    # Schema round-trip
    restored = FusedDocument.model_validate(doc.model_dump())
    assert restored.document_id == doc.document_id
    print("[OK]  Schema validation passed (model_dump -> model_validate round-trip)")
    print()


def _run_typed(text: str) -> None:
    from modules.fusion import TextFusionProcessor
    fuser = TextFusionProcessor()
    doc = fuser.fuse(typed_text=text)
    print(json.dumps(doc.model_dump(), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 4 Text Fusion verification tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--demo",  action="store_true", help="Run full synthetic demo")
    parser.add_argument("--typed", metavar="TEXT", help="Fuse a single typed text input")
    args = parser.parse_args()

    if args.demo:
        _run_demo()
    elif args.typed:
        _run_typed(args.typed)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
