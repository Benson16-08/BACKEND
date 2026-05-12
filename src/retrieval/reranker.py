from __future__ import annotations

from typing import Dict, List

from src.core.logger import logger


def rerank(
    query: str,
    vector_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Merge vector and BM25 results using Reciprocal Rank Fusion.

    Args:
        query:          Original query string (used for logging).
        vector_results: Chunks from chroma_client.query_chroma().
        bm25_results:   Chunks from bm25_index.bm25_search().
        top_k:          Final number of results to return.

    Returns:
        Deduplicated list of top_k chunks ordered by descending RRF score.
        Each chunk dict includes a 'retrieval_method' field.
    """
    K = 60

    rrf_scores: Dict[str, float] = {}
    chunk_data:  Dict[str, Dict]  = {}

    def _add_results(results: List[Dict], method: str) -> None:
        for rank, chunk in enumerate(results, start=1):
            cid = chunk.get("chunk_id", chunk.get("text", "")[:50])
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (K + rank)
            if cid not in chunk_data:
                chunk_data[cid] = {**chunk, "retrieval_method": method}
            else:
                chunk_data[cid]["retrieval_method"] = "hybrid"

    _add_results(vector_results, "vector")
    _add_results(bm25_results,   "bm25")

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    merged = []
    for cid in sorted_ids[:top_k]:
        chunk = {**chunk_data[cid], "rrf_score": round(rrf_scores[cid], 6)}
        merged.append(chunk)

    logger.info(
        f"rerank: {len(vector_results)} vector + {len(bm25_results)} BM25 "
        f"→ {len(merged)} merged (top_k={top_k})"
    )
    return merged
