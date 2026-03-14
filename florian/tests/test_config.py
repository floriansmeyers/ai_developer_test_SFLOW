from config import RETRIEVAL_TOP_K, RERANK_TOP_N, CONFIDENCE_THRESHOLD


def test_retrieval_defaults():
    assert RETRIEVAL_TOP_K == 10
    assert RERANK_TOP_N == 5


def test_confidence_threshold_is_negative():
    assert CONFIDENCE_THRESHOLD < 0
