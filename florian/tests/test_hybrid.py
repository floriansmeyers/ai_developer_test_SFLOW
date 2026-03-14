"""Tests for retrieve/hybrid.py."""

from models import Chunk, RetrievalResult
from retrieve.hybrid import reciprocal_rank_fusion


def _make_result(chunk_id: str, score: float, retriever: str = "test",
                 status: str = "current") -> RetrievalResult:
    chunk = Chunk(
        text=f"Text for {chunk_id}",
        chunk_id=chunk_id,
        source_file=f"{chunk_id}.md",
        doc_title=chunk_id,
        doc_status=status,
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=score, retriever=retriever)


def test_rrf_combines_two_lists():
    """Chunks appearing in both lists should rank higher than those in one."""
    list_a = [_make_result("a", 0.9), _make_result("b", 0.8)]
    list_b = [_make_result("a", 0.7), _make_result("c", 0.6)]
    fused = reciprocal_rank_fusion([list_a, list_b])

    ids = [r.chunk.chunk_id for r in fused]
    # "a" appears in both lists, should be first
    assert ids[0] == "a"
    assert set(ids) == {"a", "b", "c"}


def test_overlap_boosting():
    """A chunk in both retrievers should score higher than one in a single retriever."""
    list_a = [_make_result("overlap", 0.9)]
    list_b = [_make_result("overlap", 0.8)]
    list_c = [_make_result("single", 0.95)]
    fused = reciprocal_rank_fusion([list_a, list_b, list_c])
    ids = [r.chunk.chunk_id for r in fused]
    # "overlap" appears in 2 of 3 lists, should beat "single" in 1 list
    assert ids[0] == "overlap"


def test_archived_deprioritized():
    """Archived documents should score lower due to metadata weight."""
    list_a = [_make_result("current_doc", 0.9, status="current"),
              _make_result("old_doc", 0.85, status="archived")]
    fused = reciprocal_rank_fusion([list_a])
    ids = [r.chunk.chunk_id for r in fused]
    assert ids[0] == "current_doc"


def test_empty_input():
    """Empty input lists should return empty results."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order():
    """A single result list should maintain ranking order."""
    results = [_make_result("first", 0.9), _make_result("second", 0.5)]
    fused = reciprocal_rank_fusion([results])
    assert fused[0].chunk.chunk_id == "first"
    assert fused[1].chunk.chunk_id == "second"
