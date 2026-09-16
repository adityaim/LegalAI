# Legal AI — Multimodal Legal Assistant System

> **Document ingestion layer** of the Multimodal Legal Assistant System.
> Built from the system design document (`Multimodal_Legal_Assistant_System_Design.docx`).
>
> See **[WALKTHROUGH.md](WALKTHROUGH.md)** for deep-dive architecture, schemas, and design decisions.

---

## Project Status

| Module | Description | Status | Tests | Coverage | API |
|--------|-------------|:------:|:-----:|:--------:|-----|
| 1 — OCR | PDF & image → structured text | ✅ Complete | 78 | 73% | `POST /api/v1/ocr/process` |
| 2 — HTR | Handwriting region detection & recognition | ✅ Complete | 89 | 71% | `POST /api/v1/htr/process` |
| 3 — ASR | Audio / voice → transcript | 🔲 Pending | — | — | — |
| 4 — Text Fusion | Merge OCR + HTR + ASR outputs | 🔲 Pending | — | — | — |
| 5 — RAG | Legal document ingestion & retrieval | 🔲 Pending | — | — | — |
| 6 — LLM | Legal reasoning & QA | 🔲 Pending | — | — | — |
| 7 — Response | Answer generation & citation | 🔲 Pending | — | — | — |

**Combined: 167 / 167 tests passing · 71% coverage**

---

## Quick Start

```powershell
# Install base dependencies (works without GPU or heavy models)
pip install pymupdf opencv-python-headless Pillow numpy pydantic langdetect fastapi uvicorn python-multipart

# Run all tests
pip install pytest pytest-cov pytest-asyncio httpx
pytest tests/ -v --cov=modules

# Start the API server (Modules 1 + 2)
uvicorn api.main:app --reload --port 8001
# Docs: http://localhost:8001/docs
```

---

## Repository Structure

```
LegalAI/
├── modules/
│   ├── ocr/                     # Module 1: OCR Document Reader
│   │   ├── __init__.py          #   Public API
│   │   ├── reader.py            #   OCRDocumentReader (main pipeline)
│   │   ├── schema.py            #   BoundingBox, OCRBlock, OCRResult (Pydantic v2)
│   │   ├── preprocessor.py      #   Grayscale → Denoise → Binarise → Deskew
│   │   ├── layout.py            #   Heuristic layout (+ LayoutParser opt-in)
│   │   ├── postprocessor.py     #   Legal normalisation + metadata extraction
│   │   ├── utils.py             #   Skew, rotation, DPI helpers
│   │   └── engines/
│   │       ├── base.py          #   Abstract OCREngine + WordResult
│   │       ├── paddle_engine.py #   PaddleOCR adapter (primary)
│   │       └── tesseract_engine.py  # Tesseract adapter (fallback)
│   │
│   └── htr/                     # Module 2: Handwritten Document Reader
│       ├── __init__.py          #   Public API
│       ├── reader.py            #   HandwritingReader (main pipeline)
│       ├── schema.py            #   HTRRegion, HTRResult (reuses BoundingBox from M1)
│       ├── segmenter.py         #   Heuristic (OpenCV) + YOLOv8 opt-in
│       ├── postprocessor.py     #   SymSpell + legal-term guard + confidence flagging
│       └── engines/
│           ├── base.py          #   Abstract HTREngine + RegionRecognition
│           ├── trocr_engine.py  #   TrOCR (microsoft/trocr-base-handwritten)
│           └── tesseract_devanagari.py  # Tesseract 5 Hindi/Devanagari
│
├── api/
│   ├── main.py                  # FastAPI app (mounts all module routers)
│   ├── ocr_router.py            # POST /api/v1/ocr/process  GET /api/v1/ocr/health
│   └── htr_router.py            # POST /api/v1/htr/process  GET /api/v1/htr/health
│
├── tests/
│   ├── ocr/                     # 78 tests (schema, preprocessor, postprocessor, reader)
│   └── htr/                     # 89 tests (schema, segmenter, postprocessor, reader)
│
├── scripts/
│   ├── verify_ocr.py            # End-to-end OCR CLI demo tool
│   └── verify_htr.py            # End-to-end HTR CLI demo tool
│
├── requirements/
│   ├── ocr.txt                  # Module 1 dependencies
│   ├── htr.txt                  # Module 2 dependencies
│   └── dev.txt                  # Test / dev dependencies
│
├── Dockerfile.ocr               # Multi-stage Docker build
├── docker-compose.yml
├── pyproject.toml               # Project config + optional dep groups [ocr] [htr] [dev]
├── README.md                    # This file
└── WALKTHROUGH.md               # Deep-dive architecture + design decisions
```

---

## Module 1: OCR Document Reader

Converts any PDF or image to clean structured text. Uses a **hybrid strategy**: pages with a native text layer skip OCR (fast path via PyMuPDF); scanned pages go through PaddleOCR.

### What gets detected automatically
| Field | Examples |
|-------|---------|
| `doc_type` | `contract`, `affidavit`, `FIR`, `court_order`, `petition`, `deed` |
| `parties` | Contracting party names from preamble |
| `date` | Execution / signing date (ISO 8601) |
| `jurisdiction` | State, court, or city referenced |

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

### CLI

```powershell
# Built-in demo (no document needed)
python scripts/verify_ocr.py --demo

# Your own document
python scripts/verify_ocr.py --input contract.pdf --output result.json

# Hindi document
python scripts/verify_ocr.py --input hindi_contract.pdf --lang hi
```

### REST API

```powershell
# Process a document
curl -X POST http://localhost:8001/api/v1/ocr/process `
     -F "file=@contract.pdf" -F "lang=en"

# Health check
curl http://localhost:8001/api/v1/ocr/health
```

| Query Param | Default | Options |
|-------------|---------|---------|
| `lang` | `en` | `en`, `hi`, `en+hi` |
| `ocr_engine` | `paddle` | `paddle`, `tesseract` |
| `layout_engine` | `heuristic` | `heuristic`, `layoutparser` |

### Output schema (summary)

```json
{
  "document_id": "uuid4",
  "source": "OCR",
  "language": "en",
  "pages": 3,
  "full_text": "THIS AGREEMENT is entered into...",
  "blocks": [
    { "page": 1, "type": "heading", "text": "...", "confidence": 99.0,
      "bbox": {"x0":72,"y0":80,"x1":300,"y1":100,"page":1},
      "review_required": false, "ocr_engine": "pymupdf_text" }
  ],
  "metadata": { "doc_type": "contract", "parties": ["..."], "date": "2025-01-15", "jurisdiction": "Delhi" },
  "processing_time_s": 0.44,
  "low_confidence_pages": []
}
```

> `review_required` is automatically set to `true` when `confidence < 70`.

### Install

```powershell
# Base (native PDF works immediately)
pip install pymupdf opencv-python-headless Pillow numpy pydantic langdetect

# For scanned image OCR (~500 MB)
pip install paddlepaddle paddleocr
```

---

## Module 2: Handwritten Document Reader (HTR)

Detects and recognises handwritten regions in document images. Identifies four region types: **standalone notes**, **margin annotations**, **signatures**, and **table fill-ins**.

Reuses `BoundingBox` and `ImagePreprocessor` from Module 1 — no code duplication.

### Python usage

```python
from modules.htr import HandwritingReader

reader = HandwritingReader()
result = reader.process("annotated_contract.jpg")

print(result.low_confidence_regions)    # 2
print(result.flagged_regions())         # regions needing human review
print(result.to_fusion_text())          # text with [REVIEW]...[/REVIEW] markers

# Cross-module: pass a BGR numpy array directly from Module 1
result = reader.process(bgr_page_array)   # input_type = "pdf_page"
```

### CLI

```powershell
# Check what engines are available
python scripts/verify_htr.py --health

# Built-in demo (no document needed)
python scripts/verify_htr.py --demo

# Your own document
python scripts/verify_htr.py --input annotated_page.jpg --output htr_result.json

# Hindi / Devanagari handwriting
python scripts/verify_htr.py --input hindi_note.jpg --lang hi
```

### REST API

```powershell
# Process an image
curl -X POST http://localhost:8001/api/v1/htr/process `
     -F "file=@annotated_page.jpg" -F "lang=en"

# Health check
curl http://localhost:8001/api/v1/htr/health
```

| Query Param | Default | Options |
|-------------|---------|---------|
| `lang` | `en` | `en`, `hi`, `mr`, `ne` |
| `segmenter` | `heuristic` | `heuristic`, `yolov8` |
| `trocr_variant` | `base` | `base`, `large`, `small` |
| `spellcheck` | `true` | `true`, `false` |

### Output schema (summary)

```json
{
  "document_id": "uuid4",
  "source": "HTR",
  "input_type": "image",
  "language": "en",
  "regions": [
    { "page": 1, "type": "signature", "text": "R. K. Sharma",
      "confidence": 84.2, "review_required": false, "htr_engine": "trocr",
      "bbox": {"x0":165,"y0":551,"x1":395,"y1":580,"page":1} }
  ],
  "full_text": "R. K. Sharma",
  "processing_time_s": 0.40,
  "low_confidence_regions": 0
}
```

> Region types: `standalone_note` | `annotation` | `signature` | `table_fill` | `unknown`

### Install

```powershell
# Segmentation works immediately (OpenCV already installed via Module 1)

# For TrOCR text recognition (~2 GB)
pip install transformers torch

# For Hindi/Devanagari handwriting
pip install pytesseract
# + install Tesseract system package with hin language pack

# For SymSpell spell correction
pip install symspellpy
```

---

## Running the Full API

```powershell
# Start server (both Module 1 + 2 endpoints)
uvicorn api.main:app --reload --port 8001

# Browse interactive docs
start http://localhost:8001/docs

# Service root (shows all mounted modules)
curl http://localhost:8001/
```

---

## Docker

```powershell
# Build
docker build -f Dockerfile.ocr -t legal-ai .

# Run
docker run --rm -p 8001:8001 legal-ai

# Or with Docker Compose
docker-compose up
```

---

## Running Tests

```powershell
# All tests (both modules)
pytest tests/ -v --cov=modules

# Module 1 only
pytest tests/ocr/ -v --cov=modules/ocr

# Module 2 only
pytest tests/htr/ -v --cov=modules/htr
```

---

## Swapping Engines

Both modules use abstract engine interfaces — drop in your own without changing pipeline code:

```python
# Module 1: custom OCR engine
from modules.ocr.engines.base import OCREngine, WordResult

class MyOCREngine(OCREngine):
    @property
    def name(self): return "my_ocr"
    def recognize(self, image, lang="en") -> list[WordResult]: ...

reader = OCRDocumentReader(engine=MyOCREngine())

# Module 2: custom HTR engine
from modules.htr.engines.base import HTREngine, RegionRecognition

class MyHTREngine(HTREngine):
    @property
    def engine_name(self): return "my_htr"
    def recognize(self, image, lang="en") -> RegionRecognition: ...

reader = HandwritingReader(engine=MyHTREngine())
```

---

## Configuration

```python
# Module 1
from modules.ocr.reader import ReaderConfig
reader = OCRDocumentReader(ReaderConfig(
    language="hi",
    ocr_engine="paddle",
    layout_engine="heuristic",
    rasterise_dpi=300,
    min_native_chars=50,       # chars needed to skip OCR on a page
))

# Module 2
from modules.htr.reader import ReaderConfig as HTRConfig
reader = HandwritingReader(HTRConfig(
    language="en",
    segmenter_engine="heuristic",   # or "yolov8"
    trocr_variant="base",           # or "large", "small"
    enable_spellcheck=True,
    device="cpu",                   # or "cuda"
))
```

---

## Low-Confidence Handling

All modules share the same contract for flagging uncertain output:

- Any block / region with `confidence < 70` → `review_required: true`
- The downstream UI **must** highlight these in amber for human verification before they enter the RAG pipeline
- Low-confidence HTR text is wrapped in `[REVIEW]...[/REVIEW]` in `to_fusion_text()` output

---

*Last updated: 2026-09-16 — Modules 1 & 2 complete*
