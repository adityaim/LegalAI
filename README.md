# Legal AI — Multimodal Legal Assistant System

> **Full document-ingestion + retrieval pipeline** for Indian legal proceedings.  
> Built from the system design document (`Multimodal_Legal_Assistant_System_Design.docx`).  
>
> See **[WALKTHROUGH.md](WALKTHROUGH.md)** for deep-dive architecture, schemas, and design decisions per module.

---

## Project Status

| Module | Description | Status | Tests | API |
|--------|-------------|:------:|:-----:|-----|
| 1 — OCR | PDF & image → structured text | ✅ Complete | 78 | `POST /api/v1/ocr/process` |
| 2 — HTR | Handwriting region detection & recognition | ✅ Complete | 89 | `POST /api/v1/htr/process` |
| 3 — ASR | Audio / voice → transcript | ✅ Complete | 122 | `POST /api/v1/asr/process` |
| 4 — Text Fusion | Merge OCR + HTR + ASR → tagged context doc | ✅ Complete | 87 | `POST /api/v1/fusion/fuse` |
| 5 — RAG | Chunk -> embed -> store -> retrieve legal context | ✅ Complete | 73 | `POST /api/v1/rag/query` |
| 6 — LLM Legal Reasoning | Grounded legal QA with citation extraction | ✅ Complete | 68 | `POST /api/v1/llm/reason` |
| 7 — Response Generation | Format LegalAnswer → Markdown / Plain / HTML | ✅ Complete | 51 | `POST /api/v1/response/format` |

**Combined: 568 / 568 tests passing · 0 failures · Modules 1–7 complete**

---

## System Architecture

```
PDF / Image  ──►  [Module 1: OCR]  ──►  OCRResult
Handwriting  ──►  [Module 2: HTR]  ──►  HTRResult   ──►  [Module 4: Fusion]  ──►  FusedDocument
Audio        ──►  [Module 3: ASR]  ──►  ASRResult              |
Typed Query  ────────────────────────────────────────────────────┘
                                                                 |
                                                  [Module 5: RAG]
                                                  ├── Chunker (sentence-aware, 512-token windows)
                                                  ├── Embedder (sentence-transformers / Gemini)
                                                  ├── Vector Store (ChromaDB, FAISS, or mock)
                                                  └── Retriever (cosine similarity + MMR reranking)
                                                                 |
                                                  [Module 6: LLM]  (pending)
                                                  [Module 7: Response]  (pending)
```

---

## Quick Start

```powershell
# Install base dependencies (no GPU required)
pip install pymupdf opencv-python-headless Pillow numpy pydantic langdetect fastapi uvicorn python-multipart

# Run all tests (Modules 1-5)
pip install pytest pytest-cov pytest-asyncio httpx
pytest tests/ -v --cov=modules

# Start the full API server (Modules 1-5)
uvicorn api.main:app --reload --port 8001
# Interactive docs: http://localhost:8001/docs

# Quick demos (no heavy models required)
python -m scripts.verify_ocr    --demo
python -m scripts.verify_htr    --demo
python -m scripts.verify_asr    --demo
python -m scripts.verify_fusion --demo
python -m scripts.verify_rag    --demo
```

---

## Repository Structure

```
LegalAI/
├── modules/
│   ├── ocr/                     # Module 1: OCR Document Reader
│   │   ├── reader.py            #   OCRDocumentReader (main pipeline)
│   │   ├── schema.py            #   BoundingBox, OCRBlock, OCRResult (Pydantic v2)
│   │   ├── preprocessor.py      #   Grayscale -> Denoise -> Binarise -> Deskew
│   │   ├── layout.py            #   Heuristic layout (+ LayoutParser opt-in)
│   │   ├── postprocessor.py     #   Legal normalisation + metadata extraction
│   │   └── engines/             #   PaddleOCR (primary), Tesseract (fallback)
│   │
│   ├── htr/                     # Module 2: Handwritten Document Reader
│   │   ├── reader.py            #   HandwritingReader (main pipeline)
│   │   ├── schema.py            #   HTRRegion, HTRResult
│   │   ├── segmenter.py         #   OpenCV + YOLOv8 opt-in
│   │   ├── postprocessor.py     #   SymSpell + legal-term guard
│   │   └── engines/             #   TrOCR (primary), Tesseract Devanagari
│   │
│   ├── asr/                     # Module 3: ASR Voice-to-Text
│   │   ├── reader.py            #   ASRReader (main pipeline)
│   │   ├── schema.py            #   ASRSegment, ASRResult
│   │   ├── preprocessor.py      #   Audio load -> 16kHz resample -> RMS normalise
│   │   ├── postprocessor.py     #   logprob->confidence, silence filter, legal corrections
│   │   └── engines/             #   Whisper (primary), faster-whisper (opt-in)
│   │
│   ├── fusion/                  # Module 4: Text Fusion & Pre-processing
│   │   ├── processor.py         #   TextFusionProcessor (main pipeline)
│   │   ├── schema.py            #   FusedDocument, ModalityBlock, ReviewFlag
│   │   └── cleaner.py           #   NFKC clean, Jaccard dedup, review-flag extraction
│   │
│   └── rag/                     # Module 5: RAG Ingestion & Retrieval
│       ├── chunker.py           #   Sentence-aware sliding window chunker
│       ├── embedder.py          #   Pluggable embedder (sentence-transformers / Gemini)
│       ├── vector_store.py      #   Abstract store + ChromaDB / FAISS / mock adapters
│       ├── retriever.py         #   Cosine search + MMR reranking
│       ├── ingestor.py          #   RAGIngestor: FusedDocument -> vector store
│       ├── pipeline.py          #   RAGPipeline: query -> retrieved chunks
│       └── schema.py            #   Chunk, RetrievalResult, RAGQuery (Pydantic v2)
│
├── api/
│   ├── main.py                  # FastAPI app v0.5.0 (mounts all routers)
│   ├── ocr_router.py            # POST /api/v1/ocr/process
│   ├── htr_router.py            # POST /api/v1/htr/process
│   ├── asr_router.py            # POST /api/v1/asr/process
│   ├── fusion_router.py         # POST /api/v1/fusion/fuse
│   └── rag_router.py            # POST /api/v1/rag/ingest  POST /api/v1/rag/query
│
├── tests/
│   ├── ocr/    (78 tests)
│   ├── htr/    (89 tests)
│   ├── asr/    (122 tests)
│   ├── fusion/ (87 tests)
│   └── rag/    (94 tests)
│
├── scripts/
│   ├── verify_ocr.py
│   ├── verify_htr.py
│   ├── verify_asr.py
│   ├── verify_fusion.py
│   └── verify_rag.py
│
├── pyproject.toml               # Project config + optional dep groups
├── README.md
└── WALKTHROUGH.md               # Deep-dive per-module architecture docs
```

---

## Module 1: OCR Document Reader

Converts any PDF or image to clean structured text. Uses a **hybrid strategy**: pages with a native text layer skip OCR (fast via PyMuPDF); scanned pages go through PaddleOCR.

### Python usage

```python
from modules.ocr import OCRDocumentReader

reader = OCRDocumentReader()
result = reader.process("contract.pdf")

print(result.metadata.doc_type)        # "contract"
print(result.metadata.parties)         # ["Alpha Ltd", "Beta Corp"]
print(result.mean_confidence())        # 97.2
print(result.pages_needing_review())   # [2, 5]
```

### CLI / REST

```powershell
python -m scripts.verify_ocr --demo
python -m scripts.verify_ocr --input contract.pdf --output result.json

curl -X POST http://localhost:8001/api/v1/ocr/process -F "file=@contract.pdf" -F "lang=en"
curl http://localhost:8001/api/v1/ocr/health
```

| Query Param | Default | Options |
|-------------|---------|---------|
| `lang` | `en` | `en`, `hi`, `en+hi` |
| `ocr_engine` | `paddle` | `paddle`, `tesseract` |
| `layout_engine` | `heuristic` | `heuristic`, `layoutparser` |

### Output schema (abridged)

```json
{
  "document_id": "uuid4", "source": "OCR", "language": "en", "pages": 3,
  "full_text": "THIS AGREEMENT is entered into...",
  "blocks": [{ "page": 1, "type": "heading", "text": "...", "confidence": 99.0,
               "bbox": {"x0":72,"y0":80,"x1":300,"y1":100,"page":1},
               "review_required": false, "ocr_engine": "pymupdf_text" }],
  "metadata": { "doc_type": "contract", "parties": ["Alpha Ltd"], "date": "2025-01-15",
                "jurisdiction": "Delhi" },
  "processing_time_s": 0.44, "low_confidence_pages": []
}
```

### Install

```powershell
pip install pymupdf opencv-python-headless Pillow numpy pydantic langdetect
pip install paddlepaddle paddleocr   # for scanned images (~500 MB)
```

---

## Module 2: HTR Handwriting Reader

Detects and recognises handwritten regions: **standalone notes, margin annotations, signatures, table fill-ins**. Reuses `BoundingBox` and `ImagePreprocessor` from Module 1 — no code duplication.

### Python usage

```python
from modules.htr import HandwritingReader

reader = HandwritingReader()
result = reader.process("annotated_contract.jpg")

print(result.low_confidence_regions)    # 2
print(result.flagged_regions())         # regions needing human review
print(result.to_fusion_text())          # text with [REVIEW]...[/REVIEW] markers

# Cross-module: pass a BGR numpy array directly from Module 1
result = reader.process(bgr_page_array)
```

### CLI / REST

```powershell
python -m scripts.verify_htr --health
python -m scripts.verify_htr --demo
python -m scripts.verify_htr --input annotated_page.jpg --output htr_result.json

curl -X POST http://localhost:8001/api/v1/htr/process -F "file=@annotated_page.jpg"
```

### Install

```powershell
pip install transformers torch      # TrOCR recognition (~2 GB)
pip install pytesseract             # Hindi/Devanagari fallback
pip install symspellpy              # spell correction
```

---

## Module 3: ASR Voice-to-Text

Transcribes audio/voice recordings of legal proceedings. Dual-engine: **OpenAI Whisper** (default, fully offline) or **faster-whisper** (2-4x faster on CPU via CTranslate2 int8).

### Python usage

```python
from modules.asr import ASRReader
from modules.asr.reader import ReaderConfig

reader = ASRReader()
result = reader.process("court_hearing.mp3")

print(result.transcript)           # full transcript
print(result.language)             # "en" | "hi" | ...
print(result.intent)               # "question" | "instruction" | "narration"
print(result.mean_confidence())    # 82.3
print(result.to_fusion_text())     # for Module 4 (with [REVIEW] markers)

# Force Hindi + faster engine
reader = ASRReader(config=ReaderConfig(language="hi", engine_backend="faster_whisper"))
```

### CLI / REST

```powershell
python -m scripts.verify_asr --demo
python -m scripts.verify_asr --input court_hearing.mp3 --lang en

curl -X POST http://localhost:8001/api/v1/asr/process -F "file=@hearing.mp3" -F "lang=en"
```

### Install

```powershell
pip install openai-whisper soundfile scipy pydub
pip install faster-whisper          # optional CPU-optimised engine
```

---

## Module 4: Text Fusion & Pre-processing

Merges OCRResult + HTRResult + ASRResult + typed text into a single **FusedDocument** with modality-tagged blocks, cross-modality Jaccard deduplication (threshold 0.85), and `[REVIEW]` markers.

### Python usage

```python
from modules.fusion import TextFusionProcessor

fuser = TextFusionProcessor()
doc = fuser.fuse(
    ocr=ocr_result, htr=htr_result, asr=asr_result,
    typed_text="What are the payment terms?",
)

print(doc.for_rag())          # tag-stripped plain text -> Module 5
print(doc.for_llm())          # tagged text + metadata header -> Module 6
print(doc.detected_intent)    # "question"
print(doc.review_flags)       # low-confidence spans for human QA
```

### Fused text format

```
[OCR-DOCUMENT]
Alpha Technologies executed the agreement on 15 January 2025.
[/OCR-DOCUMENT]

[HTR-HANDWRITING]
Signed by R.K. Sharma. [REVIEW]see clause 4[/REVIEW]
[/HTR-HANDWRITING]

[ASR-VOICE]
What are the payment terms under Clause 1?
[/ASR-VOICE]
```

### CLI / REST

```powershell
python -m scripts.verify_fusion --demo

curl -X POST http://localhost:8001/api/v1/fusion/fuse \
     -H "Content-Type: application/json" \
     -d '{"typed_text": "Draft a notice under Section 138."}'
```

---

## Module 5: RAG Ingestion & Retrieval

Chunks `FusedDocument.for_rag()` text into overlapping windows, embeds them, stores in a vector database, and retrieves the most relevant chunks for any legal query.

### Python usage

```python
from modules.rag import RAGPipeline

pipeline = RAGPipeline()

# Ingest a fused document
pipeline.ingest(fused_doc)

# Query
results = pipeline.query("What does Section 138 NI Act say about dishonoured cheques?", top_k=5)

for chunk in results.chunks:
    print(f"[score={chunk.score:.3f}] [{chunk.source_modality}] {chunk.text[:100]}...")
```

### CLI / REST

```powershell
python -m scripts.verify_rag --demo
python -m scripts.verify_rag --query "Section 138 dishonoured cheque"

curl -X POST http://localhost:8001/api/v1/rag/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Section 138 NI Act dishonoured cheque", "top_k": 5}'
```

### Chunk schema

```json
{
  "chunk_id": "uuid4",
  "document_id": "fused-doc-uuid",
  "source_modality": "OCR",
  "text": "Alpha Technologies failed to honour the cheque...",
  "chunk_index": 3,
  "start_char": 512,
  "end_char": 1024,
  "has_review_flag": false,
  "score": 0.923
}
```

### Install

```powershell
pip install sentence-transformers chromadb   # primary embedding + vector store
pip install faiss-cpu                        # optional FAISS backend
```

---

## Running the Full API (Modules 1-5)

```powershell
uvicorn api.main:app --reload --port 8001

# Health checks
curl http://localhost:8001/
curl http://localhost:8001/api/v1/ocr/health
curl http://localhost:8001/api/v1/htr/health
curl http://localhost:8001/api/v1/asr/health
curl http://localhost:8001/api/v1/fusion/health
curl http://localhost:8001/api/v1/rag/health
```

---

## Running Tests

```powershell
# All modules
pytest tests/ -v --cov=modules

# Individual modules
pytest tests/ocr/    -v --cov=modules/ocr
pytest tests/htr/    -v --cov=modules/htr
pytest tests/asr/    -v --cov=modules/asr
pytest tests/fusion/ -v --cov=modules/fusion
pytest tests/rag/    -v --cov=modules/rag
```

---

## Docker

```powershell
docker build -f Dockerfile.ocr -t legal-ai .
docker run --rm -p 8001:8001 legal-ai
docker-compose up
```

---

## Swapping Engines

All modules use abstract engine interfaces — drop in your own without changing any pipeline code:

```python
# Module 5: custom embedder
from modules.rag.embedder import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...  # call your embedding API

pipeline = RAGPipeline(embedder=MyEmbedder())

# Module 5: custom vector store
from modules.rag.vector_store import BaseVectorStore
# implement upsert() / query() / delete_collection()
pipeline = RAGPipeline(store=MyVectorStore())
```

---

## Cross-Module Design Principles

| Principle | Implementation |
|-----------|---------------|
| Universal confidence threshold | `confidence < 70` -> `review_required: true` across all modules |
| `[REVIEW]` markers | Low-confidence spans wrapped throughout fusion + RAG |
| Pluggable engines | Abstract base class per module; inject mock in tests |
| Pydantic v2 everywhere | All inter-module contracts are typed, validated, serialisable |
| No models in CI | All heavy models mocked in tests; 0 external API calls |
| `to_fusion_text()` contract | Each modality result exposes this; Module 4 consumes uniformly |
| `for_rag()` / `for_llm()` | Module 4 exposes both; Module 5 uses `for_rag()`, M6 uses `for_llm()` |

---

## Low-Confidence Handling

```
confidence < 70  ->  review_required: true          (all modules)
                 ->  [REVIEW]text[/REVIEW]           (fusion + RAG chunks)
                 ->  surfaced in review_flags[]       (FusedDocument)
                 ->  RAG retriever down-weights       (has_review_flag: true)
```

---

*Last updated: 2026-09-23 — Modules 1-5 complete*
