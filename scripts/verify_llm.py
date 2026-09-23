"""
scripts/verify_llm.py
──────────────────────
CLI verification and demo tool for Module 6: LLM Legal Reasoning.

Usage
-----
  python -m scripts.verify_llm --demo
  python -m scripts.verify_llm --query "What is Section 138 NI Act?"
  python -m scripts.verify_llm --health
"""

from __future__ import annotations

import argparse
import json
import sys

SAMPLE_RETRIEVAL_CONTEXT = """\
[CONTEXT]

[1] source=OCR score=0.923
Section 138 of the Negotiable Instruments Act 1881 deals with dishonour of cheque for insufficiency of funds. Where any cheque drawn by a person on an account maintained by him with a banker for payment of any amount of money to another person from out of that account for the discharge of any debt or other liability is returned by the bank unpaid, the drawer shall be deemed to have committed an offence.

[2] source=ASR score=0.871
The punishment for the offence under Section 138 is imprisonment up to two years or fine up to twice the amount of the cheque or both. A legal notice must be sent within 30 days of the cheque being returned. The accused must make payment within 15 days of receipt of the notice.

[/CONTEXT]"""

SAMPLE_FUSED_TEXT = """\
[LEGAL-AI CONTEXT]
Language: en
Intent: question
Sources: OCR, ASR
Word count: 85
Review flags: 0
[/LEGAL-AI CONTEXT]

[OCR-DOCUMENT]
Section 138 of the Negotiable Instruments Act 1881 deals with dishonour of cheque for insufficiency of funds.
[/OCR-DOCUMENT]

[ASR-VOICE]
What is the punishment under Section 138?
[/ASR-VOICE]"""


def _run_demo() -> None:
    from modules.llm import LegalReasoner, LegalQuery

    print("\n=== Module 6: LLM Legal Reasoning -- Demo ===\n")

    reasoner = LegalReasoner()
    print(f"  Engine   : {reasoner.engine.engine_name}")
    print(f"  Threshold: {reasoner.config.confidence_threshold}%\n")

    queries = [
        ("Section 138 NI Act dishonoured cheque punishment", SAMPLE_RETRIEVAL_CONTEXT, SAMPLE_FUSED_TEXT),
        ("What is IPC 420 cheating?", "", ""),
        ("Explain the limitation period for filing a suit.", "", ""),
    ]

    for q, ctx, fused in queries:
        query = LegalQuery(
            query=q,
            fused_document_text=fused,
            retrieval_context=ctx,
        )
        answer = reasoner.reason(query)
        print(f"  Query      : {q[:70]!r}")
        print(f"  Engine     : {answer.llm_engine}")
        print(f"  Confidence : {answer.confidence:.1f}%")
        print(f"  Grounded   : {answer.is_grounded}")
        print(f"  Citations  : {len(answer.citations)}")
        print(f"  Review req : {answer.review_required}")
        print(f"  Time       : {answer.processing_time_s*1000:.1f} ms")
        print(f"  Answer     : {answer.answer_text[:120].replace(chr(10), ' ')}...")
        print()

    # Round-trip
    answer = reasoner.reason(LegalQuery(query="Section 138 dishonour", retrieval_context=SAMPLE_RETRIEVAL_CONTEXT))
    data = answer.model_dump()
    from modules.llm.schema import LegalAnswer
    restored = LegalAnswer.model_validate(data)
    assert restored.answer_id == answer.answer_id
    print("[OK]  Schema round-trip passed")
    print(f"[OK]  Summary: {json.dumps(answer.summary(), indent=2)}")
    print()


def _run_query(query_text: str) -> None:
    from modules.llm import LegalReasoner, LegalQuery
    reasoner = LegalReasoner()
    query = LegalQuery(query=query_text, retrieval_context=SAMPLE_RETRIEVAL_CONTEXT)
    answer = reasoner.reason(query)
    print(json.dumps(answer.model_dump(), indent=2, ensure_ascii=False))


def _run_health() -> None:
    from modules.llm import LegalReasoner
    r = LegalReasoner()
    print(f"  Engine     : {r.engine.engine_name}")
    print(f"  Threshold  : {r.config.confidence_threshold}%")
    print(f"  Max tokens : {r.config.max_tokens}")
    print("  Status     : ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 6 LLM verification tool")
    parser.add_argument("--demo",   action="store_true")
    parser.add_argument("--query",  metavar="TEXT")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    if args.demo:
        _run_demo()
    elif args.query:
        _run_query(args.query)
    elif args.health:
        _run_health()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
