from __future__ import annotations

import time
from typing import Dict, List, Optional

from src.core.exceptions import RetrievalFailedError
from src.core.logger import logger
from src.knowledge_base.kb_loader import kb


def query_chroma(
    embedding: List[float],
    n_results: int = 5,
    chapter_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Find top-n_results guideline chunks similar to query embedding.

    Args:
        embedding: 384-dim query vector from encode_query().
        n_results:      Number of chunks to return.
        chapter_filter: Optional chapter name to restrict results to.

    Returns:
        List of dicts, each with keys: text, chapter, section, chunk_id, score.

    Raises:
        RetrievalFailedError: if KB is not loaded or search fails.
    """
    if not kb.loaded:
        raise RetrievalFailedError(
            "Knowledge base is not loaded. Check startup logs."
        )

    t0 = time.perf_counter()
    try:
        raw_results = kb.vector_search(embedding, k=n_results * 2)
    except Exception as exc:
        raise RetrievalFailedError(f"Vector search failed: {exc}") from exc

    chunks = []
    for score, chunk in raw_results:
        if chapter_filter and chapter_filter.lower() not in chunk.chapter.lower():
            continue
        chunks.append({
            "text":     chunk.text,
            "chapter":  chunk.chapter,
            "section":  chunk.section,
            "chunk_id": chunk.chunk_id,
            "score":    round(score, 4),
            "source":   f"{chunk.chapter} — {chunk.section}".strip(" —"),
        })
        if len(chunks) >= n_results:
            break

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        f"query_chroma: returned {len(chunks)} chunks in {elapsed_ms}ms "
        f"(top_score={chunks[0]['score'] if chunks else 'N/A'})"
    )
    return chunks
