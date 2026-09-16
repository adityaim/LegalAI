"""
modules/htr/engines/trocr_engine.py
─────────────────────────────────────
TrOCR engine adapter — primary HTR engine for Module 2.

Uses microsoft/trocr-base-handwritten (HuggingFace Transformers).

Architecture
─────────────
TrOCR is a VisionEncoderDecoder model:
  • Encoder: ViT-Base (vision transformer) — encodes the input image
  • Decoder: RoBERTa-based autoregressive decoder — generates text tokens

Input: PIL RGB image (any size — processor resizes to 384×384 internally)
Output: decoded token sequence

Confidence scoring (design spec: "TrOCR token probability")
────────────────────────────────────────────────────────────
During generation we collect per-step logit distributions via
``output_scores=True``. For each generated token we take:
  p_t = softmax(logits_t)[chosen_token_id]

Confidence = 100 × exp( (1/T) × Σ log(p_t) )
           = 100 × geometric_mean(p_t)

This is the standard sequence-level CTC/beam confidence formulation.
Empty sequences (no text generated) get confidence = 0.

Model variants
───────────────
  trocr-base-handwritten  : ~330 MB — good balance of speed and accuracy
  trocr-large-handwritten : ~1.4 GB — highest accuracy (production default)
  trocr-small-handwritten : ~120 MB — fast, lower accuracy

Lazy loading
─────────────
Model and processor are loaded on the first recognize() call or via warmup().
This avoids loading 330 MB at import time.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from .base import HTREngine, RegionRecognition

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Model identifiers
TROCR_BASE = "microsoft/trocr-base-handwritten"
TROCR_LARGE = "microsoft/trocr-large-handwritten"
TROCR_SMALL = "microsoft/trocr-small-handwritten"

# Generation config
MAX_NEW_TOKENS = 128    # max tokens per handwritten region
NUM_BEAMS = 4           # beam search width (higher = more accurate, slower)


class TrOCREngine(HTREngine):
    """
    HTR engine backed by Microsoft TrOCR (transformers library).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        ``"cpu"`` or ``"cuda"``. Defaults to CPU for development portability.
    num_beams : int
        Beam search width. Higher = more accurate but slower.
    """

    def __init__(
        self,
        model_name: str = TROCR_BASE,
        device: str = "cpu",
        num_beams: int = NUM_BEAMS,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._num_beams = num_beams
        self._processor = None
        self._model = None
        self._available: bool | None = None

    # ── Public interface ───────────────────────────────────────────────────

    @property
    def engine_name(self) -> str:
        return "trocr"

    @property
    def supported_languages(self) -> list[str]:
        # TrOCR base/large handwritten is English-focused
        # Multilingual TrOCR variants exist but are not yet official
        return ["en"]

    def warmup(self) -> None:
        """Load model and processor eagerly."""
        self._load_model()
        logger.info("TrOCR engine warmed up (%s)", self._model_name)

    def recognize(self, image: np.ndarray, lang: str = "en") -> RegionRecognition:
        """
        Run TrOCR on a single handwritten region crop.

        Returns an empty RegionRecognition (confidence=0) if the model is
        not available or if recognition produces no output.
        """
        if not self._load_model():
            return RegionRecognition(text="", confidence=0.0, language=lang)

        pil_img = self._to_pil_rgb(image)
        if pil_img is None:
            return RegionRecognition(text="", confidence=0.0, language=lang)

        try:
            import torch
            pixel_values = self._processor(
                images=pil_img, return_tensors="pt"
            ).pixel_values.to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    pixel_values,
                    num_beams=self._num_beams,
                    max_new_tokens=MAX_NEW_TOKENS,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = outputs.sequences
            scores = outputs.scores  # tuple of (vocab_size,) tensors per step

            text = self._processor.batch_decode(
                sequences, skip_special_tokens=True
            )[0].strip()

            confidence = self._compute_confidence(sequences, scores)

        except Exception as exc:
            logger.error("TrOCR inference error: %s", exc)
            return RegionRecognition(text="", confidence=0.0, language=lang)

        return RegionRecognition(text=text, confidence=confidence, language=lang)

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        """Load processor + model lazily. Returns True if available."""
        if self._available is False:
            return False
        if self._processor is not None:
            return True

        try:
            from transformers import (  # type: ignore[import]
                TrOCRProcessor,
                VisionEncoderDecoderModel,
            )
            logger.info("Loading TrOCR model: %s (device=%s)", self._model_name, self._device)
            self._processor = TrOCRProcessor.from_pretrained(self._model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self._model_name)
            self._model = self._model.to(self._device)
            self._model.eval()
            self._available = True
            logger.info("TrOCR model loaded successfully")
            return True

        except ImportError:
            logger.warning(
                "transformers / torch not installed — TrOCR unavailable. "
                "Install with: pip install transformers torch"
            )
            self._available = False
            return False
        except Exception as exc:
            logger.error("Failed to load TrOCR model '%s': %s", self._model_name, exc)
            self._available = False
            return False

    def _compute_confidence(self, sequences, scores) -> float:
        """
        Compute geometric-mean token probability as a confidence score (0–100).

        ``sequences`` shape: (batch=1, seq_len)
        ``scores``    : tuple of length seq_len, each (batch=1, vocab_size)

        We skip the first generated token (BOS/EOS artefact) and compute:
            confidence = 100 × exp( mean( log( softmax(s)[token_id] ) ) )
        """
        if not scores:
            return 0.0

        try:
            import torch
            import torch.nn.functional as F

            token_ids = sequences[0, 1:]  # skip BOS token at position 0
            log_probs: list[float] = []

            for step_idx, (step_scores, token_id) in enumerate(
                zip(scores, token_ids)
            ):
                if step_idx >= len(token_ids):
                    break
                probs = F.softmax(step_scores[0], dim=-1)
                token_prob = probs[token_id].item()
                if token_prob > 0:
                    log_probs.append(math.log(token_prob))

            if not log_probs:
                return 0.0

            geo_mean = math.exp(sum(log_probs) / len(log_probs))
            return round(min(100.0, geo_mean * 100.0), 2)

        except Exception as exc:
            logger.debug("Confidence computation failed: %s", exc)
            return 0.0

    @staticmethod
    def _to_pil_rgb(image: np.ndarray):  # noqa: ANN205
        """Convert numpy array (BGR or gray) to PIL RGB image for TrOCR processor."""
        try:
            from PIL import Image
            import cv2

            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return Image.fromarray(rgb)
        except Exception as exc:
            logger.error("Image conversion failed: %s", exc)
            return None
