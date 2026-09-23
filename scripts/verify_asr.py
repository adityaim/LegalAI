"""
scripts/verify_asr.py
──────────────────────
CLI verification and demo tool for Module 3: ASR Voice-to-Text.

Usage
─────
  # Synthetic demo (generates a silent WAV, tests the full pipeline with a mock engine)
  python scripts/verify_asr.py --demo

  # Transcribe a real audio file
  python scripts/verify_asr.py --input court_hearing.mp3 --output result.json

  # Force Hindi, use faster-whisper
  python scripts/verify_asr.py --input deposition.wav --lang hi --engine faster_whisper

  # Health check (print available dependencies)
  python scripts/verify_asr.py --health

  # REST API (server must be running on port 8001)
  curl -X POST http://localhost:8001/api/v1/asr/process -F "file=@hearing.mp3"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ─── Health check ─────────────────────────────────────────────────────────────

def _check_health() -> None:
    print("\n=== ASR Module — Dependency Health Check ===\n")
    deps = {
        "openai-whisper":    ("whisper",        "pip install openai-whisper"),
        "faster-whisper":    ("faster_whisper",  "pip install faster-whisper"),
        "soundfile":         ("soundfile",       "pip install soundfile"),
        "pydub":             ("pydub",           "pip install pydub"),
        "scipy":             ("scipy",           "pip install scipy"),
        "torch":             ("torch",           "pip install torch"),
        "numpy":             ("numpy",           "pip install numpy"),
    }
    all_ok = True
    for name, (pkg, install_cmd) in deps.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  [OK] {name:<22} v{ver}")
        except ImportError:
            print(f"  [--] {name:<22} NOT installed  ({install_cmd})")
            if name in ("openai-whisper", "soundfile", "numpy"):
                all_ok = False
    print()
    if all_ok:
        print("[OK]  All required dependencies present.")
    else:
        print("[WARN] Some required dependencies are missing.")
    print()


# ─── Demo (mock engine, synthetic audio) ─────────────────────────────────────

def _run_demo() -> None:
    """
    Run the full pipeline with a mock ASR engine to verify schema,
    postprocessor, and reader — without requiring Whisper to be installed.
    """
    print("\n=== ASR Module — Demo (Mock Engine) ===\n")

    from modules.asr.engines.base import ASREngine, SegmentResult, TranscriptionResult
    from modules.asr.reader import ASRReader, ReaderConfig
    from modules.asr.schema import ASRResult

    class MockASREngine(ASREngine):
        """Synthetic engine that returns hard-coded legal transcript segments."""

        def transcribe(
            self,
            audio: np.ndarray,
            language: str | None = None,
            initial_prompt: str | None = None,
        ) -> TranscriptionResult:
            return TranscriptionResult(
                segments=[
                    SegmentResult(
                        segment_id=0, start_s=0.0, end_s=4.2,
                        text="The plaintiff filed a case under Section 138 of the NI Act.",
                        avg_logprob=-0.3, no_speech_prob=0.02, language="en",
                    ),
                    SegmentResult(
                        segment_id=1, start_s=4.2, end_s=9.1,
                        text="The accused failed to appear before the High Court on the scheduled date.",
                        avg_logprob=-0.8, no_speech_prob=0.05, language="en",
                    ),
                    SegmentResult(
                        segment_id=2, start_s=9.1, end_s=13.5,
                        text="Draft a legal notice under IPC Section 420 for the respondent.",
                        avg_logprob=-0.5, no_speech_prob=0.03, language="en",
                    ),
                    SegmentResult(
                        segment_id=3, start_s=13.5, end_s=18.0,
                        text="garbled inaudible noise here",
                        avg_logprob=-3.8, no_speech_prob=0.08, language="en",
                    ),
                ],
                language="en",
                language_probability=0.98,
                duration_s=18.0,
            )

        @property
        def engine_name(self) -> str:
            return "mock"

    # Create synthetic silent audio (1 second) — only used for duration measurement
    silent_audio = np.zeros(16_000, dtype=np.float32)

    config = ReaderConfig(language="en")
    reader = ASRReader(config=config, engine=MockASREngine())

    t0 = time.perf_counter()
    result: ASRResult = reader.process(silent_audio)
    elapsed = time.perf_counter() - t0

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"  Segments found  : {len(result.segments)}")
    print(f"  Language        : {result.language} (prob={result.language_probability:.2f})")
    print(f"  Intent          : {result.intent}")
    print(f"  Mean confidence : {result.mean_confidence():.1f}%")
    print(f"  Low-conf segs   : {result.low_confidence_segments}")
    print(f"  Processing      : {elapsed:.3f}s")
    print()
    print("  Transcript:")
    print("  " + "-" * 60)
    for seg in result.segments:
        flag = " [REVIEW]" if seg.review_required else ""
        print(
            f"  [{seg.start_s:.1f}s–{seg.end_s:.1f}s] "
            f"conf={seg.confidence:.0f}%{flag}"
        )
        print(f"    {seg.text}")
    print()
    print("  Module 4 fusion text:")
    print("  " + "-" * 60)
    print("  " + result.to_fusion_text().replace("\n", "\n  "))
    print()

    # Schema validation
    data = result.model_dump()
    restored = type(result).model_validate(data)
    assert restored.document_id == result.document_id, "Round-trip mismatch!"
    print("[OK]  Schema validation passed (model_dump -> model_validate round-trip)")
    print()


# ─── Real file processing ─────────────────────────────────────────────────────

def _run_file(
    input_path: str,
    output_path: str | None,
    lang: str | None,
    engine_backend: str,
    model_size: str,
) -> None:
    from modules.asr.reader import ASRReader, ReaderConfig

    path = Path(input_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing: {path.name}")
    config = ReaderConfig(
        language=lang or None,
        engine_backend=engine_backend,      # type: ignore[arg-type]
        model_size=model_size,
    )
    reader = ASRReader(config=config)

    t0 = time.perf_counter()
    result = reader.process(path)
    elapsed = time.perf_counter() - t0

    print(f"\n  Duration        : {result.duration_s:.1f}s")
    print(f"  Language        : {result.language} (prob={result.language_probability:.2f})")
    print(f"  Intent          : {result.intent}")
    print(f"  Segments        : {len(result.segments)}")
    print(f"  Mean confidence : {result.mean_confidence():.1f}%")
    print(f"  Low-conf segs   : {result.low_confidence_segments}")
    print(f"  Processing time : {elapsed:.2f}s")
    print()
    print("  Transcript (truncated to 500 chars):")
    print("  " + result.transcript[:500])
    print()

    if output_path:
        out = Path(output_path)
        out.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        print(f"[OK]  Result saved to: {out}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 3 ASR verification tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--demo",   action="store_true", help="Run synthetic demo (mock engine)")
    parser.add_argument("--health", action="store_true", help="Print dependency health check")
    parser.add_argument("--input",  metavar="FILE",   help="Audio file to transcribe")
    parser.add_argument("--output", metavar="FILE",   help="Save result JSON to this file")
    parser.add_argument("--lang",   metavar="LANG",   help="ISO 639-1 language code (default: auto)")
    parser.add_argument("--engine", metavar="ENGINE", default="whisper",
                        choices=["whisper", "faster_whisper"],
                        help="ASR engine backend (default: whisper)")
    parser.add_argument("--model",  metavar="SIZE",   default="base",
                        help="Whisper model size (default: base)")
    args = parser.parse_args()

    if args.health:
        _check_health()
    elif args.demo:
        _run_demo()
    elif args.input:
        _run_file(args.input, args.output, args.lang, args.engine, args.model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
