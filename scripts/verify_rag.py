"""
scripts/verify_rag.py
──────────────────────
CLI verification and demo tool for Module 5: RAG Ingestion & Retrieval.

Usage
-----
  # Full demo: ingest 3 synthetic legal documents, run 3 queries
  python -m scripts.verify_rag --demo

  # Query the current in-memory store (after --demo populates it)
  python -m scripts.verify_rag --query "Section 138 dishonoured cheque"

  # Health
  python -m scripts.verify_rag --health
"""

from __future__ import annotations

import argparse
import json
import sys


# ─── Synthetic fused docs ─────────────────────────────────────────────────────

def _make_fused_doc(title: str, content: str, modality: str = "OCR") -> object:
    """Return a duck-typed object that satisfies FusedDocument protocol."""

    class _FakeFusedDoc:
        def __init__(self):
            import uuid
            self.document_id = str(uuid.uuid4())
            self.fused_text = f"[{modality}-DOCUMENT]\n{content}\n[/{modality}-DOCUMENT]" \
                if modality == "OCR" else f"[ASR-VOICE]\n{content}\n[/ASR-VOICE]"

        def for_rag(self) -> str:
            return self.fused_text \
                .replace("[OCR-DOCUMENT]\n", "").replace("\n[/OCR-DOCUMENT]", "") \
                .replace("[ASR-VOICE]\n", "").replace("\n[/ASR-VOICE]", "")

    return _FakeFusedDoc()


DEMO_DOCS = [
    (
        "NI Act Section 138",
        (
            "Section 138 of the Negotiable Instruments Act 1881 deals with dishonour of cheque "
            "for insufficiency of funds. Where any cheque drawn by a person on an account maintained "
            "by him with a banker for payment of any amount of money to another person from out of "
            "that account for the discharge of any debt or other liability is returned by the bank "
            "unpaid, the drawer shall be deemed to have committed an offence. The offence is "
            "punishable with imprisonment up to two years or fine up to twice the amount of the "
            "cheque or both. A legal notice must be sent within 30 days of cheque dishonour."
        ),
        "OCR",
    ),
    (
        "IPC Section 420 Cheating",
        (
            "Section 420 of the Indian Penal Code deals with cheating and dishonestly inducing "
            "delivery of property. Whoever cheats and thereby dishonestly induces the person "
            "deceived to deliver any property to any person, or to make, alter or destroy "
            "the whole or any part of a valuable security, shall be punished with imprisonment "
            "of either description for a term which may extend to seven years. "
            "The Supreme Court has held that mere breach of contract does not amount to cheating "
            "under IPC Section 420 unless fraudulent intent is established from the inception."
        ),
        "OCR",
    ),
    (
        "High Court Hearing Transcript",
        (
            "The plaintiff appeared before the High Court and submitted that the accused had "
            "issued a cheque for Rupees five lakhs. The cheque was presented and returned unpaid "
            "by the bank with the remark 'insufficient funds'. A legal notice was duly served "
            "within the prescribed period of 30 days. The accused failed to make payment within "
            "15 days of receipt of the legal notice. The court admitted the complaint under "
            "Section 138 of the Negotiable Instruments Act."
        ),
        "ASR",
    ),
]

DEMO_QUERIES = [
    "What happens when a cheque is dishonoured due to insufficient funds?",
    "What is the punishment under Section 138 NI Act?",
    "How many days does one have to send a legal notice after cheque dishonour?",
]


def _run_demo() -> None:
    from modules.rag import RAGPipeline
    from modules.rag.schema import RetrievalResult

    print("\n=== Module 5: RAG Ingestion & Retrieval -- Demo ===\n")
    pipeline = RAGPipeline()

    # Ingest
    total_chunks = 0
    for title, content, modality in DEMO_DOCS:
        doc = _make_fused_doc(title, content, modality)
        chunks = pipeline.ingest(doc)
        total_chunks += len(chunks)
        print(f"  [Ingested] '{title}' -> {len(chunks)} chunks")

    print(f"\n  Collection size : {pipeline.collection_size()} chunks total")
    print()

    # Query
    for q in DEMO_QUERIES:
        result: RetrievalResult = pipeline.query(q, top_k=3)
        print(f"  Query: {q!r}")
        print(f"  Retrieved {len(result.chunks)} chunks in {result.retrieval_time_s*1000:.1f} ms")
        for i, chunk in enumerate(result.chunks, 1):
            score = f"{chunk.score:.3f}" if chunk.score is not None else "n/a"
            print(f"    [{i}] score={score} [{chunk.source_modality}] {chunk.preview(80)}")
        print()

    # Schema round-trip
    result = pipeline.query("Section 138 legal notice", top_k=2)
    data = result.model_dump()
    from modules.rag.schema import RetrievalResult as RR
    restored = RR.model_validate(data)
    assert restored.query == result.query
    print("[OK]  Schema round-trip (model_dump -> model_validate) passed")
    print("[OK]  context_for_llm snippet:")
    print("  " + result.context_for_llm[:200].replace("\n", "\n  "))
    print()


def _run_query(query: str) -> None:
    from modules.rag import RAGPipeline
    pipeline = RAGPipeline()
    result = pipeline.query(query)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


def _run_health() -> None:
    from modules.rag import RAGPipeline
    pipeline = RAGPipeline()
    print(f"  Embedder     : {pipeline.embedder.model_name}")
    print(f"  Vector store : {type(pipeline.store).__name__}")
    print(f"  Collection   : {pipeline.config.collection}")
    print(f"  Chunk count  : {pipeline.collection_size()}")
    print("  Status       : ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 5 RAG verification tool")
    parser.add_argument("--demo",   action="store_true", help="Run full ingestion + query demo")
    parser.add_argument("--query",  metavar="TEXT", help="Run a single query")
    parser.add_argument("--health", action="store_true", help="Health check")
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
