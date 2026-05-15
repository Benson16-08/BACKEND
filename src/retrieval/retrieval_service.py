"""
src/retrieval/retrieval_service.py
────────────────────────────────────
Main retrieval entrypoint — called by the query route handler.

Pipeline:
  1. encode_query()     →  384-dim query vector  (blocking — runs in threadpool)
  2. query_chroma()     →  top-k vector results  (blocking — runs in threadpool)
  3. bm25_search()      →  top-k keyword results (blocking — runs in threadpool)
  4. rerank()           →  fused top-k chunks    (CPU-only, fast)
  5. token_budget_guard →  ensures total tokens ≤ RETRIEVAL_TOKEN_BUDGET

All blocking calls run inside asyncio.get_event_loop().run_in_executor()
so they never block the async event loop. Two simultaneous requests to
/api/v1/query will both be served concurrently.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from src.core.config import settings
from src.core.exceptions import RetrievalFailedError
from src.core.logger import logger
from src.retrieval.bm25_index import bm25_search
from src.retrieval.chroma_client import query_chroma
from src.retrieval.embedder import encode_query
from src.retrieval.reranker import rerank


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _apply_token_budget(chunks: List[Dict], budget: int) -> List[Dict]:
    """
    Trim chunk list so total token count stays within budget.
    Removes from the end (lowest-ranked first).
    """
    total = 0
    kept = []
    for chunk in chunks:
        tokens = _estimate_tokens(chunk.get("text", ""))
        if total + tokens > budget:
            logger.warning(
                f"Token budget ({budget}) reached after {len(kept)} chunks — "
                f"dropping {len(chunks) - len(kept)} lower-ranked chunks."
            )
            break
        total += tokens
        kept.append(chunk)
    return kept


def _encode_and_search(query: str, k: int) -> tuple:
    """
    Blocking function: encode query + run both searches.
    Designed to run inside run_in_executor() — never call directly from async.
    Returns (embedding, vector_results, bm25_results).
    """
    embedding = encode_query(query)
    vector_results = query_chroma(embedding, n_results=k * 2)
    bm25_results = bm25_search(query, k=k * 2)
    return embedding, vector_results, bm25_results


async def retrieve_chunks(
    query: str,
    k: Optional[int] = None,
) -> List[Dict]:
    """
    Full async hybrid retrieval pipeline.

    All blocking I/O runs in a threadpool via run_in_executor so the
    FastAPI event loop is never blocked. Two simultaneous requests are
    handled concurrently.

    Args:
        query: Symptom description from the doctor.
        k:     Number of chunks to return (defaults to settings.RETRIEVAL_TOP_K).

    Returns:
        List of up to k chunk dicts with: text, chapter, section,
        chunk_id, source, rrf_score, retrieval_method.

    Raises:
        RetrievalFailedError: on any failure in the pipeline.
    """
    if not query or not query.strip():
        raise RetrievalFailedError("Query cannot be empty.")

    k = k or settings.RETRIEVAL_TOP_K
    t0 = time.perf_counter()

    try:
        loop = asyncio.get_event_loop()

        # Run all blocking retrieval work in a single threadpool call
        t_search = time.perf_counter()
        embedding, vector_results, bm25_results = await loop.run_in_executor(
            None, _encode_and_search, query, k
        )
        search_ms = int((time.perf_counter() - t_search) * 1000)

        # Rerank is CPU-only and fast — runs in the event loop directly
        merged = rerank(query, vector_results, bm25_results, top_k=k)

        # Token budget guard
        final = _apply_token_budget(merged, settings.RETRIEVAL_TOKEN_BUDGET)

        total_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            f"retrieve_chunks: query_len={len(query)}  "
            f"search_ms={search_ms}  total_ms={total_ms}  "
            f"vector={len(vector_results)}  bm25={len(bm25_results)}  "
            f"returned={len(final)}"
        )
        return final

    except RetrievalFailedError:
        raise
    except Exception as exc:
        logger.error(f"retrieve_chunks unexpected error: {exc}")
        raise RetrievalFailedError(f"Retrieval pipeline failed: {exc}") from exc
