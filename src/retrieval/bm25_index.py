from __future__ import annotations

import re
import time
from typing import Dict, List

from rank_bm25 import BM25Okapi

from src.core.logger import logger
from src.knowledge_base.kb_loader import kb


def _tokenise(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]", " ", text.lower()).split()


# Lazy initialization
_TOKENISED_CORPUS: List[List[str]] | None = None
_BM25: BM25Okapi | None = None


def _ensure_bm25_index():
    """Build BM25 index if not already built."""
    global _TOKENISED_CORPUS, _BM25
    
    if _BM25 is not None:
        return
    
    if not kb._loaded:
        kb.load()
    
    logger.info("Building BM25 index from KB texts...")
    t0 = time.perf_counter()
    
    _TOKENISED_CORPUS = [_tokenise(c.text) for c in kb.chunks]
    _BM25 = BM25Okapi(_TOKENISED_CORPUS)
    
    bm25_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"BM25 index built: {len(_TOKENISED_CORPUS)} docs in {bm25_ms}ms ✓")


def bm25_search(query: str, k: int = 5) -> List[Dict]:
    """
    Keyword search over all STG guideline chunks using BM25Okapi.

    Args:
        query: Free-text symptom query string.
        k:     Number of top results to return.

    Returns:
        List of dicts with keys: text, chapter, section, chunk_id, score, source.
        Ordered by descending BM25 score. Excludes zero-score results.
    """
    _ensure_bm25_index()
    
    if not query.strip():
        return []

    tokens = _tokenise(query)
    scores = _BM25.get_scores(tokens)

    import numpy as np
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            break
        chunk = kb.chunks[idx]
        results.append({
            "text":     chunk.text,
            "chapter":  chunk.chapter,
            "section":  chunk.section,
            "chunk_id": chunk.chunk_id,
            "score":    round(score, 4),
            "source":   f"{chunk.chapter} — {chunk.section}".strip(" —"),
        })

    return results
