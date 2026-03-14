import config
from models import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = config.RRF_K,
) -> list[RetrievalResult]:
    """Merge multiple ranked result lists using RRF.

    For each chunk, the RRF score is: sum(1 / (k + rank_i)) across all
    retrievers where the chunk appears.
    """
    # Map chunk_id -> (best RetrievalResult, cumulative RRF score)
    scores: dict[str, float] = {}
    best_result: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            cid = result.chunk.chunk_id
            rrf_score = 1.0 / (k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + rrf_score

            if cid not in best_result or result.score > best_result[cid].score:
                best_result[cid] = result

    # Apply metadata weight
    weighted: list[tuple[str, float]] = []
    for cid, score in scores.items():
        doc_status = best_result[cid].chunk.doc_status
        weight = config.METADATA_WEIGHTS.get(doc_status, 1.0)
        weighted.append((cid, score * weight))

    # Sort by weighted RRF score
    weighted.sort(key=lambda x: x[1], reverse=True)

    return [
        RetrievalResult(
            chunk=best_result[cid].chunk,
            score=score,
            retriever="hybrid",
        )
        for cid, score in weighted
    ]
