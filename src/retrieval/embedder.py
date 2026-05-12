from __future__ import annotations

import os
import time
import warnings
from typing import List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

from src.core.config import settings
from src.core.logger import logger

logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL} ...")
_t0 = time.perf_counter()

try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)
    _MODEL_AVAILABLE = True
    _load_ms = int((time.perf_counter() - _t0) * 1000)
    logger.info(f"Embedding model loaded in {_load_ms}ms ✓")
except Exception as exc:
    _MODEL = None
    _MODEL_AVAILABLE = False
    logger.warning(
        f"Embedding model could not load ({exc}). "
        "Falling back to KB-vector search via pre-stored embeddings."
    )


def encode_query(text: str) -> List[float]:
    """
    Encode a symptom query into a 384-dim embedding vector.
    Falls back to zeros (BM25-only mode) if model is unavailable.
    """
    if not text or not text.strip():
        raise ValueError("Cannot encode an empty query string.")

    if _MODEL_AVAILABLE and _MODEL is not None:
        vec: List[float] = _MODEL.encode(
            text.strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        return vec
    else:
        logger.warning("encode_query: model unavailable, returning zero vector (BM25 mode)")
        return [0.0] * settings.EMBEDDING_DIMENSION


def is_model_available() -> bool:
    return _MODEL_AVAILABLE
