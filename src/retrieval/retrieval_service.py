from __future__ import annotations

import asyncio
import sys
import time
from typing import Dict, List
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



from src.core.config import settings
from src.core.exceptions import RetrievalFailedError
from src.core.logger import logger
from src.knowledge_base.kb_loader import kb
from src.retrieval.bm25_index import bm25_search
from src.retrieval.chroma_client import query_chroma
from src.retrieval.embedder import encode_query
from src.retrieval.reranker import rerank


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters (GPT-style)."""
    return len(text) // 4


def _apply_token_budget(chunks: List[Dict], budget: int) -> List[Dict]:
    """
    Trim chunk list so total token count does not exceed budget.
    Trims from the end (lowest-ranked chunks removed first).
    """
    total = 0
    kept = []
    for chunk in chunks:
        tokens = _estimate_tokens(chunk.get("text", ""))
        if total + tokens > budget:
            logger.warning(
                f"Token budget ({budget}) reached after {len(kept)} chunks. "
                f"Dropping remaining {len(chunks) - len(kept)} chunks."
            )
            break
        total += tokens
        kept.append(chunk)
    return kept


async def retrieve_chunks(
    query: str,
    k: int | None = None,
) -> List[Dict]:
    """
    Full hybrid retrieval pipeline: encode → vector search → BM25 → rerank → budget.

    Args:
        query: Symptom description from the doctor.
        k:     Number of chunks to return (defaults to settings.RETRIEVAL_TOP_K).

    Returns:
        List of up to k chunk dicts, each containing:
          text, chapter, section, chunk_id, source, rrf_score, retrieval_method.

    Raises:
        RetrievalFailedError: on any failure in the pipeline.
    """
    if not query or not query.strip():
        raise RetrievalFailedError("Query cannot be empty.")

    k = k or settings.RETRIEVAL_TOP_K
    kb.load()  # Ensure knowledge base is loaded
    t0 = time.perf_counter()

    try:
        t_enc = time.perf_counter()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, encode_query, query)
        enc_ms = int((time.perf_counter() - t_enc) * 1000)

        t_vec = time.perf_counter()
        vector_results = query_chroma(embedding, n_results=k * 2)
        vec_ms = int((time.perf_counter() - t_vec) * 1000)

        t_bm25 = time.perf_counter()
        bm25_results = bm25_search(query, k=k * 2)
        bm25_ms = int((time.perf_counter() - t_bm25) * 1000)

        merged = rerank(query, vector_results, bm25_results, top_k=k)

        final = _apply_token_budget(merged, settings.RETRIEVAL_TOKEN_BUDGET)

        total_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            f"retrieve_chunks: query_len={len(query)}  "
            f"enc={enc_ms}ms  vec={vec_ms}ms  bm25={bm25_ms}ms  "
            f"total={total_ms}ms  returned={len(final)} chunks"
        )
        return final

    except RetrievalFailedError:
        raise
    except Exception as exc:
        logger.error(f"retrieve_chunks unexpected error: {exc}")
        raise RetrievalFailedError(f"Retrieval pipeline failed: {exc}") from exc
