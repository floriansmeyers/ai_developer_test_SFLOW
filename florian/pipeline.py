"""End-to-end RAG pipeline: ingest, index, retrieve, generate."""

import logging

import config
from models import RetrievalResult
from ingest.loader import load_documents
from ingest.chunker import chunk_documents
from index.vector_store import index_chunks, search_vectors, collection_exists
from index.bm25_store import BM25Index
from retrieve.hybrid import reciprocal_rank_fusion
from retrieve.reranker import rerank
from generate.prompt import SYSTEM_PROMPT, build_user_prompt
from generate.llm import generate_answer, strip_sources_block

logger = logging.getLogger(__name__)

_MAX_QUESTION_LENGTH = 500


class RAGPipeline:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._bm25: BM25Index | None = None
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
        )

    def build_index(self, force: bool = False) -> None:
        documents = load_documents(config.DOCS_DIR)
        for doc in documents:
            logger.debug("  [%8s] %s", doc.doc_status, doc.source_file)

        chunks = chunk_documents(documents, max_words=config.CHUNK_MAX_WORDS)
        logger.debug("Created %d chunks from %d documents.", len(chunks), len(documents))

        logger.debug("Building vector index (embedding with OpenAI)...")
        index_chunks(chunks)

        logger.debug("Building BM25 index...")
        self._bm25 = BM25Index(chunks)
        self._bm25.save()

        logger.debug("Indexing complete.")

    def retrieve(self, question: str) -> list[RetrievalResult]:
        if self._bm25 is None:
            self._bm25 = BM25Index.load()

        top_k = config.RETRIEVAL_TOP_K
        rerank_top_n = config.RERANK_TOP_N

        # Dual retrieval
        vector_results = search_vectors(question, top_k=top_k)
        bm25_results = self._bm25.search(question, top_k=top_k)

        logger.debug("Vector results: %d", len(vector_results))
        for r in vector_results[:3]:
            logger.debug("  %s (score: %.3f)", r.chunk.source_file, r.score)
        logger.debug("BM25 results: %d", len(bm25_results))
        for r in bm25_results[:3]:
            logger.debug("  %s (score: %.3f)", r.chunk.source_file, r.score)

        # Hybrid fusion
        hybrid_results = reciprocal_rank_fusion([vector_results, bm25_results])

        logger.debug("Hybrid results (after RRF): %d", len(hybrid_results))
        for r in hybrid_results[:5]:
            logger.debug("  %s [%s] (score: %.4f)", r.chunk.source_file, r.chunk.doc_status, r.score)

        # Re-rank
        top_results = rerank(question, hybrid_results, top_n=rerank_top_n)

        logger.debug("After re-ranking: %d", len(top_results))
        for r in top_results:
            logger.debug("  %s [%s] (score: %.4f)", r.chunk.source_file, r.chunk.doc_status, r.score)

        return top_results

    def answer(self, question: str) -> tuple[str, list[RetrievalResult]]:
        # Input sanitization: truncate overly long questions
        if len(question) > _MAX_QUESTION_LENGTH:
            logger.warning("Question truncated from %d to %d characters.", len(question), _MAX_QUESTION_LENGTH)
            question = question[:_MAX_QUESTION_LENGTH]

        try:
            results = self.retrieve(question)

            # No results at all — refuse without LLM call
            if not results:
                return (
                    "The provided documentation does not contain sufficient information "
                    "to answer this question.",
                    [],
                )

            # Confidence gate: if top reranker score is too low, refuse without LLM call
            if results[0].score < config.CONFIDENCE_THRESHOLD:
                logger.debug(
                    "Low confidence (%.3f < %.1f). Returning canned refusal.",
                    results[0].score, config.CONFIDENCE_THRESHOLD,
                )
                return (
                    "The provided documentation does not contain sufficient information "
                    "to answer this question.",
                    results,
                )

            user_prompt = build_user_prompt(question, results)
            response = generate_answer(SYSTEM_PROMPT, user_prompt)

            # Strip any LLM-generated Sources: block (CLI shows deterministic panel)
            response = strip_sources_block(response)

            return response, results

        except Exception:
            logger.exception("Error answering question: %s", question[:100])
            return (
                "An error occurred while processing your question. Please try again.",
                [],
            )
