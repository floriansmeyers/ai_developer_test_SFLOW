import logging

import config
from models import RetrievalResult

_logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank(
    query: str,
    results: list[RetrievalResult],
    top_n: int = config.RERANK_TOP_N,
) -> list[RetrievalResult]:
    """Re-rank results using a cross-encoder model.

    Falls back to truncating the input list if cross-encoder is disabled or fails.
    """
    if not config.USE_CROSS_ENCODER:
        return results[:top_n]

    try:
        model = _get_cross_encoder()
        pairs = [(query, r.chunk.text) for r in results]
        scores = model.predict(pairs)

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                chunk=r.chunk,
                score=float(s),
                retriever="reranked",
            )
            for r, s in scored[:top_n]
        ]
    except Exception:
        _logger.warning("Cross-encoder reranking failed; falling back to RRF order", exc_info=True)
        return results[:top_n]
