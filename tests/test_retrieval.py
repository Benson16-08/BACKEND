import pytest
import asyncio

from src.retrieval.embedder import encode_query, is_model_available
from src.retrieval.chroma_client import query_chroma
from src.retrieval.bm25_index import bm25_search
from src.retrieval.reranker import rerank
from src.retrieval.retrieval_service import retrieve_chunks


class TestEmbedder:

    def test_encode_query_returns_list(self, kb_loaded):
        vec = encode_query("fever and headache")
        assert isinstance(vec, list)

    def test_encode_query_correct_dimension(self, kb_loaded):
        vec = encode_query("fever and headache")
        assert len(vec) == 384

    def test_encode_query_all_floats(self, kb_loaded):
        vec = encode_query("malaria treatment guidelines")
        assert all(isinstance(v, float) for v in vec)

    def test_encode_empty_query_raises(self, kb_loaded):
        with pytest.raises(ValueError):
            encode_query("")

    def test_encode_whitespace_raises(self, kb_loaded):
        with pytest.raises(ValueError):
            encode_query("   ")

    def test_second_call_returns_same_dimension(self, kb_loaded):
        v1 = encode_query("fever")
        v2 = encode_query("pneumonia in children")
        assert len(v1) == len(v2) == 384


class TestChromaClient:

    def test_query_chroma_returns_list(self, kb_loaded):
        vec = encode_query("fever and chills")
        results = query_chroma(vec, n_results=5)
        assert isinstance(results, list)

    def test_query_chroma_returns_correct_count(self, kb_loaded):
        vec = encode_query("malaria")
        results = query_chroma(vec, n_results=3)
        assert len(results) == 3

    def test_query_chroma_each_result_has_text(self, kb_loaded):
        vec = encode_query("fever")
        results = query_chroma(vec, n_results=5)
        for r in results:
            assert "text" in r
            assert len(r["text"]) > 0

    def test_query_chroma_each_result_has_metadata(self, kb_loaded):
        vec = encode_query("malaria treatment")
        results = query_chroma(vec, n_results=5)
        for r in results:
            assert "chapter" in r
            assert "section" in r
            assert "chunk_id" in r

    def test_query_chroma_has_score(self, kb_loaded):
        vec = encode_query("fever")
        results = query_chroma(vec, n_results=3)
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)


class TestBM25:

    def test_bm25_search_returns_list(self, kb_loaded):
        results = bm25_search("malaria fever chills", k=5)
        assert isinstance(results, list)

    def test_bm25_search_finds_malaria(self, kb_loaded):
        results = bm25_search("malaria fever chills", k=5)
        assert len(results) >= 1
        all_text = " ".join(r["text"] for r in results).lower()
        assert "malaria" in all_text

    def test_bm25_search_finds_pneumonia(self, kb_loaded):
        results = bm25_search("pneumonia children cough", k=5)
        assert len(results) >= 1

    def test_bm25_search_empty_returns_empty(self, kb_loaded):
        results = bm25_search("", k=5)
        assert results == []

    def test_bm25_results_have_required_fields(self, kb_loaded):
        results = bm25_search("fever headache", k=3)
        for r in results:
            assert "text" in r
            assert "chapter" in r
            assert "section" in r
            assert "chunk_id" in r
            assert "score" in r

    def test_bm25_scores_positive(self, kb_loaded):
        results = bm25_search("malaria", k=5)
        for r in results:
            assert r["score"] > 0


class TestReranker:

    def test_rerank_returns_list(self, kb_loaded):
        vec = encode_query("fever")
        v_results = query_chroma(vec, n_results=5)
        b_results = bm25_search("fever", k=5)
        merged = rerank("fever", v_results, b_results, top_k=5)
        assert isinstance(merged, list)

    def test_rerank_deduplicates_by_chunk_id(self, kb_loaded):
        vec = encode_query("malaria")
        v_results = query_chroma(vec, n_results=10)
        b_results = bm25_search("malaria", k=10)
        merged = rerank("malaria", v_results, b_results, top_k=5)
        chunk_ids = [m["chunk_id"] for m in merged]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids in reranked results"

    def test_rerank_respects_top_k(self, kb_loaded):
        vec = encode_query("fever")
        v_results = query_chroma(vec, n_results=10)
        b_results = bm25_search("fever", k=10)
        merged = rerank("fever", v_results, b_results, top_k=3)
        assert len(merged) <= 3

    def test_rerank_results_have_rrf_score(self, kb_loaded):
        vec = encode_query("fever")
        v_results = query_chroma(vec, n_results=5)
        b_results = bm25_search("fever", k=5)
        merged = rerank("fever", v_results, b_results, top_k=5)
        for m in merged:
            assert "rrf_score" in m
            assert m["rrf_score"] > 0

    def test_rerank_with_empty_bm25(self, kb_loaded):
        vec = encode_query("fever")
        v_results = query_chroma(vec, n_results=5)
        merged = rerank("fever", v_results, [], top_k=5)
        assert len(merged) >= 1

    def test_rerank_with_empty_vector(self, kb_loaded):
        b_results = bm25_search("malaria", k=5)
        merged = rerank("malaria", [], b_results, top_k=5)
        assert len(merged) >= 1


class TestRetrievalService:

    def test_retrieve_chunks_returns_list(self, kb_loaded):
        results = asyncio.run(retrieve_chunks("fever and headache"))
        assert isinstance(results, list)

    def test_retrieve_chunks_returns_up_to_5(self, kb_loaded):
        results = asyncio.run(retrieve_chunks("fever and headache"))
        assert 1 <= len(results) <= 5

    def test_retrieve_chunks_within_token_budget(self, kb_loaded):
        results = asyncio.run(retrieve_chunks("fever with chills and headache"))
        total_tokens = sum(len(r["text"]) // 4 for r in results)
        assert total_tokens <= 4000, f"Token budget exceeded: {total_tokens}"

    def test_retrieve_chunks_each_has_text(self, kb_loaded):
        results = asyncio.run(retrieve_chunks("malaria treatment"))
        for r in results:
            assert "text" in r and len(r["text"]) > 0

    def test_retrieve_chunks_each_has_metadata(self, kb_loaded):
        results = asyncio.run(retrieve_chunks("malaria"))
        for r in results:
            assert "chapter" in r
            assert "section" in r
            assert "chunk_id" in r

    def test_retrieve_chunks_empty_query_raises(self, kb_loaded):
        from src.core.exceptions import RetrievalFailedError

        with pytest.raises(RetrievalFailedError):
            asyncio.run(retrieve_chunks(""))

    def test_retrieve_chunks_golden_query_malaria(self, kb_loaded):
        results = asyncio.run(retrieve_chunks("Treatment for severe malaria"))
        all_text = " ".join(r["text"] + r["section"] for r in results).lower()
        assert "malaria" in all_text

    def test_retrieve_chunks_golden_query_pneumonia(self, kb_loaded):
        results = asyncio.run(retrieve_chunks("Management of pneumonia in children"))
        all_text = " ".join(r["text"] + r["section"] for r in results).lower()
        assert "pneumonia" in all_text or "respiratory" in all_text
