"""
scripts/verify_response.py
───────────────────────────
CLI verification and demo tool for Module 7: Response Generation & Formatting.

Usage
-----
  python -m scripts.verify_response --demo
  python -m scripts.verify_response --health
"""

from __future__ import annotations

import argparse
import sys

from modules.llm.schema import Citation, LegalAnswer
from modules.response import ResponseFormatter
from modules.response.schema import ResponseFormat


def _make_sample_answer() -> LegalAnswer:
    return LegalAnswer(
        query="What is the punishment under Section 138 of the NI Act?",
        answer_text=(
            "Section 138 of the Negotiable Instruments Act, 1881 provides that where a cheque "
            "is dishonoured due to insufficiency of funds, the drawer shall be deemed to have "
            "committed an offence [1]. The punishment prescribed is imprisonment up to two years, "
            "or a fine up to twice the amount of the cheque, or both [1][2]. "
            "[REVIEW]The exact quantum of fine is at the court's discretion.[/REVIEW] "
            "A legal notice must be sent within 30 days of the cheque being returned [2]."
        ),
        citations=[
            Citation(
                chunk_id="chunk-001",
                document_id="doc-001",
                source_modality="OCR",
                text_excerpt="Section 138 of the Negotiable Instruments Act deals with dishonour of cheque.",
                relevance_score=0.923,
            ),
            Citation(
                chunk_id="chunk-002",
                document_id="doc-001",
                source_modality="ASR",
                text_excerpt="Punishment is imprisonment up to two years or fine up to twice the cheque amount.",
                relevance_score=0.871,
            ),
        ],
        confidence=85.0,
        language="en",
        detected_intent="question",
        processing_time_s=0.43,
        llm_engine="mock",
        is_grounded=True,
    )


def _run_demo() -> None:
    print("\n=== Module 7: Response Generation -- Demo ===\n")

    answer = _make_sample_answer()
    formatter = ResponseFormatter()

    for fmt in [ResponseFormat.MARKDOWN, ResponseFormat.PLAIN, ResponseFormat.HTML, ResponseFormat.JSON]:
        response = formatter.format(answer, fmt=fmt)
        print(f"--- FORMAT: {fmt.value.upper()} ---")
        if fmt == ResponseFormat.JSON:
            import json
            print(json.dumps(response.to_json_summary(), indent=2))
        elif fmt == ResponseFormat.HTML:
            print(response.to_html()[:400] + "...")
        elif fmt == ResponseFormat.MARKDOWN:
            print(response.to_markdown()[:500] + "...")
        else:
            print(response.to_plain()[:400] + "...")
        print()

    # Verify marker stripping
    assert "[REVIEW]" not in formatter.format(answer).answer
    print("[OK]  [REVIEW] markers stripped from formatted answer")

    # Verify citations
    response = formatter.format(answer)
    assert len(response.citations) == 2
    assert response.citations[0].relevance_score >= response.citations[1].relevance_score
    print(f"[OK]  {len(response.citations)} citations, sorted by score")
    print(f"[OK]  Response ID: {response.response_id}")
    print()


def _run_health() -> None:
    formatter = ResponseFormatter()
    print(f"  Default format     : {formatter.config.default_format.value}")
    print(f"  Max citations      : {formatter.config.max_citations}")
    print(f"  Strip markers      : {formatter.config.strip_internal_markers}")
    print("  Status             : ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 7 Response verification tool")
    parser.add_argument("--demo",   action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    if args.demo:
        _run_demo()
    elif args.health:
        _run_health()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
