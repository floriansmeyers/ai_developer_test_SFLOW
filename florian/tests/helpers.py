"""Shared test helpers."""

import hashlib


def fake_embeddings(texts: list[str]) -> list[list[float]]:
    """Deterministic 128-d embeddings for testing."""
    result = []
    for text in texts:
        h = hashlib.md5(text.encode()).hexdigest()
        emb = [int(h[i : i + 2], 16) / 255.0 for i in range(0, 32, 2)]
        emb.extend([0.0] * (128 - len(emb)))
        norm = sum(x * x for x in emb) ** 0.5
        if norm > 0:
            emb = [x / norm for x in emb]
        result.append(emb)
    return result
