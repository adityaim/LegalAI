# Legal AI — Module 1: OCR Document Reader

> Part of the **Multimodal Legal Assistant System** (Section 4.1 of the system design document).

## What This Module Does

Converts scanned legal PDFs and document images into structured, clean text with bounding boxes, confidence scores, and lightweight document metadata.

**Hybrid PDF strategy:** pages with a native text layer skip OCR entirely (PyMuPDF direct extraction). Only truly scanned pages go through the OCR pipeline. This achieves >2 pages/s throughput (NFR-2).

## Quick Start

```bash
# 1. Install dependencies (CPU-only)
pip install -r requirements/ocr.txt

# 2. Verify the module with a synthetic demo PDF
python scripts/verify_ocr.py --demo

# 3. Process a real document
python scripts/verify_ocr.py --input path/to/contract.pdf --output result.json

# 4. Run tests
pip install -r requirements/dev.txt
pytest tests/ocr/ -v --cov=modules/ocr
```

## Architecture

```
OCRDocumentReader.process(path)
├── PDF input
│   ├── PyMuPDF: has text layer? ──→ native text blocks (confidence=99, engine=pymupdf_text)
│   └── Scanned page ────────────→ rasterise@300DPI → [preprocess → layout → OCR]
└── Image input (JPEG/PNG/TIFF/WebP)
    └── [preprocess → layout → OCR]

Preprocess: grayscale → NLM denoise → Otsu binarise → Hough deskew
Layout:     heuristic (OpenCV morphology) │ LayoutParser (opt-in, GPU)
OCR:        PaddleOCR (primary) │ Tesseract (optional fallback)
Post:       legal normalisation → confidence flagging → metadata extraction
```

## Output Schema

```json
{
  "document_id": "uuid4",
  "source": "OCR",
  "language": "en",
  "pages": 3,
  "full_text": "THIS AGREEMENT is entered into...",
  "blocks": [
    {
      "page": 1, "type": "heading", "text": "THIS AGREEMENT",
      "confidence": 98.2, "bbox": {"x0":72,"y0":80,"x1":300,"y1":100,"page":1},
      "review_required": false, "ocr_engine": "pymupdf_text"
    },
    {
      "page": 2, "type": "body", "text": "...",
      "confidence": 55.0, "review_required": true, "ocr_engine": "paddle"
    }
  ],
  "metadata": {
    "doc_type": "contract",
    "parties": ["Alpha Ltd", "Beta Corp"],
    "date": "2025-01-15",
    "jurisdiction": "Delhi"
  },
  "processing_time_s": 1.24,
  "low_confidence_pages": [2]
}
```

## Module Structure

```
modules/ocr/
├── __init__.py          # Public API
├── reader.py            # OCRDocumentReader (main pipeline)
├── preprocessor.py      # Image pre-processing pipeline
├── layout.py            # Layout block detection (heuristic + LayoutParser)
├── postprocessor.py     # Legal text normalisation + metadata extraction
├── schema.py            # Pydantic v2 output schema
├── utils.py             # Image helpers (skew, rotation, DPI)
└── engines/
    ├── base.py           # Abstract OCREngine + WordResult
    ├── paddle_engine.py  # PaddleOCR adapter (primary)
    └── tesseract_engine.py # Tesseract adapter (optional fallback)
```

## FastAPI Service

```bash
# Run the API server
uvicorn api.main:app --reload --port 8001

# Process a document via API
curl -X POST http://localhost:8001/api/v1/ocr/process \
     -F "file=@contract.pdf" \
     -F "lang=en"

# Health check
curl http://localhost:8001/api/v1/ocr/health
```

### Query Parameters for `/api/v1/ocr/process`

| Parameter | Default | Options |
|-----------|---------|---------|
| `lang` | `en` | `en`, `hi`, `en+hi`, etc. |
| `ocr_engine` | `paddle` | `paddle`, `tesseract` |
| `layout_engine` | `heuristic` | `heuristic`, `layoutparser` |

## Docker

```bash
# Build
docker build -f Dockerfile.ocr -t legal-ai-ocr .

# Run
docker run --rm -p 8001:8001 legal-ai-ocr

# Or with Docker Compose
docker-compose up ocr
```

## Configuration

```python
from modules.ocr import OCRDocumentReader
from modules.ocr.reader import ReaderConfig

reader = OCRDocumentReader(ReaderConfig(
    language="hi",              # Hindi primary language
    ocr_engine="paddle",        # PaddleOCR (default)
    layout_engine="heuristic",  # fast CPU layout detection
    rasterise_dpi=300,
    min_native_chars=50,        # chars before skipping OCR
))
result = reader.process("document.pdf")
```

## Swapping the OCR Engine

The `OCREngine` interface lets you inject any backend:

```python
from modules.ocr.engines.base import OCREngine, WordResult
import numpy as np

class MyCustomEngine(OCREngine):
    @property
    def name(self): return "my_engine"
    def recognize(self, image: np.ndarray, lang="en") -> list[WordResult]:
        ...  # your implementation

reader = OCRDocumentReader(engine=MyCustomEngine())
```

## Low-Confidence Handling

- Blocks with `confidence < 70` → `review_required: True`
- Pages where mean confidence < 70 → listed in `low_confidence_pages`
- The downstream UI must highlight `review_required=True` blocks for human review before they enter the RAG pipeline

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Primary OCR engine | PaddleOCR | Superior Hindi/Devanagari; better on degraded scans |
| Layout default | Heuristic (OpenCV) | No GPU/Detectron2 required; <50ms/page CPU |
| PDF handling | PyMuPDF first | 10-100× faster for native-text PDFs |
| Language detection | langdetect | Lightweight; works on first 2000 chars |
| Confidence threshold | 70.0 | Matches design doc spec for `review_required` |
