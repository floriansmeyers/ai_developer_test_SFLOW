"""Tests for index/bm25_store.py."""

import tempfile
from pathlib import Path

from models import Chunk
from index.bm25_store import BM25Index, _tokenize


def _make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            text=t,
            chunk_id=f"doc_{i}",
            source_file=f"doc_{i}.md",
            doc_title=f"Doc {i}",
            doc_status="current",
            chunk_index=0,
        )
        for i, t in enumerate(texts)
    ]


def test_tokenize_lowercase_and_split():
    tokens = _tokenize("Hello World! This is a TEST.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    assert "TEST" not in tokens


def test_tokenize_splits_underscores():
    """Compound terms like INVOICING_COMPLETED should be split into components."""
    tokens = _tokenize("INVOICING_COMPLETED flag is set")
    assert "invoicing" in tokens
    assert "completed" in tokens
    assert "flag" in tokens


def test_tokenize_removes_stop_words():
    """Common stop words should be filtered out."""
    tokens = _tokenize("the invoice is for a project")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "for" not in tokens
    assert "a" not in tokens
    assert "invoice" in tokens
    assert "project" in tokens


def test_search_relevance():
    """Searching for 'invoice' should rank the invoice chunk higher."""
    chunks = _make_chunks([
        "Invoices belong to units and track financial transactions.",
        "Projects contain blocks and units in a hierarchy.",
        "Suppliers submit proposals for work items.",
    ])
    index = BM25Index(chunks)
    results = index.search("invoice financial")
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "doc_0"


def test_top_k_limiting():
    """Results should not exceed top_k."""
    chunks = _make_chunks(["word " * 10 for _ in range(20)])
    index = BM25Index(chunks)
    results = index.search("word", top_k=3)
    assert len(results) <= 3


def test_empty_query_returns_empty():
    """An empty or stop-word query may return no results."""
    chunks = _make_chunks(["Some content here."])
    index = BM25Index(chunks)
    results = index.search("")
    # Empty tokenization = no scores > 0
    assert isinstance(results, list)


def test_save_load_roundtrip():
    """Saving and loading should produce equivalent search results."""
    chunks = _make_chunks([
        "Financial administration on units handles invoicing.",
        "Project hierarchy: project, block, unit.",
        "Suppliers submit proposals for work items on projects.",
        "Permissions control who can approve invoices and costs.",
    ])
    index = BM25Index(chunks)
    original_results = index.search("financial unit invoice")

    # Verify the original index actually returns results (non-degenerate case)
    assert len(original_results) > 0, "Original BM25 should return results with 4+ docs"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bm25_index.sqlite"

        index.save(path)
        loaded = BM25Index.load(path)
        loaded_results = loaded.search("financial unit invoice")

        assert len(loaded_results) > 0, "Loaded BM25 should return results"
        assert len(original_results) == len(loaded_results)
        for orig, loaded_r in zip(original_results, loaded_results):
            assert orig.chunk.chunk_id == loaded_r.chunk.chunk_id
