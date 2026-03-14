import logging

import chromadb
from chromadb.errors import InvalidCollectionException, NotFoundError

import config
from models import Chunk, DocStatus, RetrievalResult
from index.embeddings import embed_texts, embed_query

COLLECTION_NAME = "easify_docs"

_chroma_client = None

logger = logging.getLogger(__name__)

# Silence noisy ChromaDB telemetry errors
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma_client


def _chunk_metadata(chunk: Chunk) -> dict:
    return {
        "source_file": chunk.source_file,
        "doc_title": chunk.doc_title,
        "doc_status": chunk.doc_status,
        "chunk_index": chunk.chunk_index,
    }


def index_chunks(chunks: list[Chunk]) -> None:
    client = _get_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, NotFoundError):
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [_chunk_metadata(c) for c in chunks]

    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.debug("Index built with %d chunks.", len(chunks))


def search_vectors(query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[RetrievalResult]:
    client = _get_client()
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieval_results = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        chunk = Chunk(
            text=results["documents"][0][i],
            chunk_id=results["ids"][0][i],
            source_file=meta["source_file"],
            doc_title=meta["doc_title"],
            doc_status=DocStatus(meta["doc_status"]),
            chunk_index=meta["chunk_index"],
        )
        # ChromaDB returns cosine distance; convert to similarity
        score = 1.0 - results["distances"][0][i]
        retrieval_results.append(RetrievalResult(chunk=chunk, score=score, retriever="vector"))

    return retrieval_results


def collection_exists() -> bool:
    client = _get_client()
    try:
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except (ValueError, NotFoundError, InvalidCollectionException):
        return False


def get_collection_count() -> int:
    client = _get_client()
    try:
        col = client.get_collection(COLLECTION_NAME)
        return col.count()
    except (ValueError, NotFoundError, InvalidCollectionException):
        return 0
