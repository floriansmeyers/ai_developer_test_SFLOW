import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import chromadb

from tests.helpers import fake_embeddings


def _setup_docs_dir(base: str) -> Path:
    docs = Path(base) / "files"
    docs.mkdir()
    (docs / "invoicing.md").write_text(
        "# Invoice Management\n\n"
        "Invoices are created per unit. Each invoice tracks costs and payments. "
        "The status field can be DRAFT, SUBMITTED, or APPROVED."
    )
    (docs / "projects.md").write_text(
        "# Project Hierarchy\n\n"
        "Projects contain blocks. Blocks contain units. "
        "Units are the lowest level where costs are tracked."
    )
    (docs / "archived_old_process.md").write_text(
        "ARCHIVED DOCUMENT\n\n"
        "This describes the old invoice process that is no longer used."
    )
    return docs


class TestFullPipeline:

    def setup_method(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.docs_dir = _setup_docs_dir(self.tmpdir.name)
        self.chroma_client = chromadb.EphemeralClient()
        self.project_root = Path(self.tmpdir.name)

        self._patches = [
            patch("config.DOCS_DIR", self.docs_dir),
            patch("config.CHROMA_DIR", self.project_root / "chroma"),
            patch("config.BM25_INDEX_PATH", self.project_root / "bm25.sqlite"),
            patch("config.PROJECT_ROOT", self.project_root),
            patch("index.vector_store._get_client", return_value=self.chroma_client),
            patch("index.vector_store.embed_texts", side_effect=fake_embeddings),
            patch("index.vector_store.embed_query", side_effect=lambda q: fake_embeddings([q])[0]),
            patch("config.USE_CROSS_ENCODER", False),
        ]
        for p in self._patches:
            p.start()

    def teardown_method(self):
        for p in reversed(self._patches):
            p.stop()
        self.tmpdir.cleanup()

    def test_build_index_creates_bm25_artifact(self):
        from pipeline import RAGPipeline

        pipeline = RAGPipeline(verbose=False)
        pipeline.build_index(force=True)

        bm25_path = self.project_root / "bm25.sqlite"
        assert bm25_path.exists()

    def test_build_index_populates_vector_store(self):
        from pipeline import RAGPipeline
        from index.vector_store import get_collection_count

        pipeline = RAGPipeline(verbose=False)
        pipeline.build_index(force=True)

        assert get_collection_count() > 0

    def test_retrieve_returns_relevant_sources(self):
        from pipeline import RAGPipeline

        pipeline = RAGPipeline(verbose=False)
        pipeline.build_index(force=True)

        results = pipeline.retrieve("How do invoices work?")
        assert len(results) > 0
        sources = [r.chunk.source_file for r in results]
        assert any("invoicing" in s for s in sources)

    def test_archived_doc_classified_through_pipeline(self):
        from pipeline import RAGPipeline

        pipeline = RAGPipeline(verbose=False)
        pipeline.build_index(force=True)

        results = pipeline.retrieve("old invoice process")
        archived = [r for r in results if r.chunk.doc_status == "archived"]
        assert len(archived) > 0

    def test_full_answer_with_mocked_llm(self):
        from pipeline import RAGPipeline

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Invoices are created per unit and track costs and payments."
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("config.get_openai_client", return_value=mock_client), \
             patch("config.CONFIDENCE_THRESHOLD", -999):
            pipeline = RAGPipeline(verbose=False)
            pipeline.build_index(force=True)

            answer, results = pipeline.answer("How do invoices work?")

            assert "invoic" in answer.lower()
            assert len(results) > 0
            mock_client.chat.completions.create.assert_called_once()

    def test_metadata_weights_order_results(self):
        from pipeline import RAGPipeline

        pipeline = RAGPipeline(verbose=False)
        pipeline.build_index(force=True)

        results = pipeline.retrieve("invoice")
        # Results should be ordered by score (descending)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_reindex_rebuilds(self):
        from pipeline import RAGPipeline
        from index.vector_store import get_collection_count

        pipeline = RAGPipeline(verbose=False)
        pipeline.build_index(force=True)
        count1 = get_collection_count()

        pipeline.build_index(force=True)
        count2 = get_collection_count()

        assert count1 == count2

    def test_empty_retrieval_returns_refusal_without_llm_call(self):
        from pipeline import RAGPipeline

        mock_client = MagicMock()

        with patch("config.get_openai_client", return_value=mock_client):
            pipeline = RAGPipeline(verbose=False)
            pipeline.build_index(force=True)

            # Mock retrieve to return empty results
            with patch.object(pipeline, "retrieve", return_value=[]):
                answer, results = pipeline.answer("Something completely unrelated?")

            assert "does not contain sufficient information" in answer
            assert results == []
            mock_client.chat.completions.create.assert_not_called()

    def test_low_confidence_returns_refusal_without_llm_call(self):
        from pipeline import RAGPipeline

        mock_client = MagicMock()

        with patch("config.get_openai_client", return_value=mock_client), \
             patch("config.CONFIDENCE_THRESHOLD", 999):
            pipeline = RAGPipeline(verbose=False)
            pipeline.build_index(force=True)

            answer, results = pipeline.answer("How do invoices work?")

            assert "does not contain sufficient information" in answer
            assert len(results) > 0
            mock_client.chat.completions.create.assert_not_called()

    def test_rebuild_picks_up_new_documents(self):
        from pipeline import RAGPipeline

        pipeline = RAGPipeline(verbose=False)
        pipeline.build_index(force=True)

        # Add a new document
        (self.docs_dir / "expenses.md").write_text(
            "# Expense Reports\n\n"
            "Employees submit expense reports for reimbursement. "
            "Each report must include receipts and manager approval."
        )

        # Rebuild picks up the new doc
        pipeline.build_index()

        results = pipeline.retrieve("expense reports reimbursement")
        sources = [r.chunk.source_file for r in results]
        assert any("expenses" in s for s in sources)
