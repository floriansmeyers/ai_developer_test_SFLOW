from unittest.mock import patch

import chromadb

from models import Chunk, DocStatus
from index.vector_store import (
    _chunk_metadata,
    COLLECTION_NAME,
)
from tests.helpers import fake_embeddings


def _make_chunk(chunk_id: str, text: str = "test text", status: str = "current") -> Chunk:
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        source_file=f"{chunk_id}.md",
        doc_title=f"Title {chunk_id}",
        doc_status=status,
        chunk_index=0,
    )


def test_chunk_metadata_fields():
    chunk = Chunk(
        text="x", chunk_id="id", source_file="src.md",
        doc_title="Title", doc_status="archived", chunk_index=3,
    )
    meta = _chunk_metadata(chunk)
    assert meta == {
        "source_file": "src.md",
        "doc_title": "Title",
        "doc_status": "archived",
        "chunk_index": 3,
    }


class TestIndexAndSearch:
    def setup_method(self):
        self.client = chromadb.EphemeralClient()
        self._patches = [
            patch("index.vector_store.embed_texts", side_effect=fake_embeddings),
            patch("index.vector_store.embed_query", side_effect=lambda q: fake_embeddings([q])[0]),
            patch("index.vector_store._get_client", return_value=self.client),
        ]
        for p in self._patches:
            p.start()

    def teardown_method(self):
        for p in reversed(self._patches):
            p.stop()

    def test_index_and_search(self):
        from index.vector_store import index_chunks, search_vectors, collection_exists, get_collection_count

        chunks = [
            _make_chunk("finance", "Financial administration handles invoicing and payments."),
            _make_chunk("project", "Projects contain blocks and units in a hierarchy."),
            _make_chunk("supplier", "Suppliers submit proposals for work items."),
        ]
        index_chunks(chunks)

        assert collection_exists()
        assert get_collection_count() == 3

        results = search_vectors("financial invoicing", top_k=2)
        assert len(results) == 2
        assert all(r.retriever == "vector" for r in results)

    def test_collection_exists_when_empty(self):
        from index.vector_store import collection_exists
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        assert not collection_exists()

    def test_get_collection_count_when_missing(self):
        from index.vector_store import get_collection_count
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        assert get_collection_count() == 0

    def test_metadata_stored_correctly(self):
        from index.vector_store import index_chunks

        chunk = _make_chunk("meta_test", "Metadata storage test.")
        chunk.doc_status = "archived"
        chunk.chunk_index = 3
        index_chunks([chunk])

        collection = self.client.get_collection(COLLECTION_NAME)
        result = collection.get(ids=["meta_test"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["source_file"] == "meta_test.md"
        assert meta["doc_status"] == "archived"
        assert meta["chunk_index"] == 3

    def test_search_returns_correct_metadata(self):
        from index.vector_store import index_chunks, search_vectors

        chunk = Chunk(
            text="Unique searchable content for testing metadata roundtrip.",
            chunk_id="unique_doc",
            source_file="special/path.md",
            doc_title="Special Title",
            doc_status="informal",
            chunk_index=7,
        )
        index_chunks([chunk])

        results = search_vectors("Unique searchable content", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert r.chunk.source_file == "special/path.md"
        assert r.chunk.doc_title == "Special Title"
        assert r.chunk.doc_status == DocStatus.INFORMAL
        assert r.chunk.chunk_index == 7

    def test_cosine_similarity_score_range(self):
        from index.vector_store import index_chunks, search_vectors

        index_chunks([_make_chunk("doc", "Test document content.")])
        results = search_vectors("Test document content.", top_k=1)
        assert len(results) == 1
        # Cosine similarity should be approximately between -1 and 1
        assert -1.01 <= results[0].score <= 1.01
