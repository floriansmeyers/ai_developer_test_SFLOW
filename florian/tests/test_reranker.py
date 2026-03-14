"""Tests for retrieve/reranker.py."""

from unittest.mock import patch

from models import Chunk, RetrievalResult


def _make_result(chunk_id: str, score: float) -> RetrievalResult:
    chunk = Chunk(
        text=f"Text for {chunk_id}",
        chunk_id=chunk_id,
        source_file=f"{chunk_id}.md",
        doc_title=chunk_id,
        doc_status="current",
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=score, retriever="hybrid")


def test_fallback_truncation_when_disabled():
    """With cross-encoder disabled, rerank should just truncate the list."""
    with patch("retrieve.reranker.config") as mock_config:
        mock_config.USE_CROSS_ENCODER = False
        mock_config.RERANK_TOP_N = 3

        from retrieve.reranker import rerank
        results = [_make_result(f"doc{i}", 1.0 - i * 0.1) for i in range(10)]
        reranked = rerank("test query", results, top_n=3)

        assert len(reranked) == 3
        assert reranked[0].chunk.chunk_id == "doc0"


def test_fallback_with_fewer_results():
    """When fewer results than top_n, return all of them."""
    with patch("retrieve.reranker.config") as mock_config:
        mock_config.USE_CROSS_ENCODER = False
        mock_config.RERANK_TOP_N = 5

        from retrieve.reranker import rerank
        results = [_make_result("only", 0.9)]
        reranked = rerank("test query", results, top_n=5)

        assert len(reranked) == 1


def test_graceful_fallback_when_cross_encoder_fails():
    """When cross-encoder raises, rerank falls back to RRF order."""
    with patch("retrieve.reranker.config") as mock_config, \
         patch("retrieve.reranker._get_cross_encoder", side_effect=OSError("model download failed")):
        mock_config.USE_CROSS_ENCODER = True
        mock_config.RERANK_TOP_N = 2

        from retrieve.reranker import rerank
        results = [_make_result(f"doc{i}", 1.0 - i * 0.1) for i in range(5)]
        reranked = rerank("test query", results, top_n=2)

        assert len(reranked) == 2
        assert reranked[0].chunk.chunk_id == "doc0"
        assert reranked[1].chunk.chunk_id == "doc1"
