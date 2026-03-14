"""Tests for ingest/chunker.py."""

from models import Document
from ingest.chunker import chunk_documents, _split_by_sections, _split_oversized, _add_overlap, CHUNK_HARD_CAP_WORDS, CHUNK_OVERLAP_WORDS


def _make_doc(content: str, source="test.md", title="Test", status="current") -> Document:
    return Document(content=content, source_file=source, doc_title=title, doc_status=status)


def test_small_doc_stays_whole():
    """Documents under max_words should produce exactly one chunk."""
    doc = _make_doc("This is a short document with only a few words.")
    chunks = chunk_documents([doc], max_words=200)
    assert len(chunks) == 1
    assert chunks[0].text == doc.content


def test_large_doc_gets_split():
    """Documents over max_words should be split into multiple chunks."""
    content = "## Section One\n\nFirst section text.\n\n## Section Two\n\nSecond section text."
    doc = _make_doc(content)
    chunks = chunk_documents([doc], max_words=5)
    assert len(chunks) > 1


def test_hard_cap_enforced():
    """No chunk should exceed the hard cap word count, even after title/overlap."""
    long_section = " ".join(["word"] * 500)
    doc = _make_doc(long_section)
    chunks = chunk_documents([doc], max_words=50)
    for chunk in chunks:
        assert len(chunk.text.split()) <= CHUNK_HARD_CAP_WORDS


def test_overlap_between_chunks():
    """Consecutive chunks from a split doc should have overlap text."""
    content = "## A\n\n" + " ".join(["alpha"] * 100) + "\n\n## B\n\n" + " ".join(["beta"] * 100)
    doc = _make_doc(content)
    chunks = chunk_documents([doc], max_words=50)
    if len(chunks) >= 2:
        # Second chunk should start with "..." (overlap marker)
        assert "..." in chunks[1].text


def test_title_prepended_on_split():
    """When a doc is split, each chunk should have the doc title prepended."""
    content = "## First\n\nSome text here.\n\n## Second\n\nMore text here."
    doc = _make_doc(content, title="My Doc")
    chunks = chunk_documents([doc], max_words=5)
    for chunk in chunks:
        assert "[My Doc]" in chunk.text


def test_chunk_ids_are_unique():
    """All chunk IDs should be unique across the output."""
    doc = _make_doc("## A\n\nText.\n\n## B\n\nText.\n\n## C\n\nText.")
    chunks = chunk_documents([doc], max_words=3)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
