import re
import sqlite3

from rank_bm25 import BM25Okapi

ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven",
    "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every",
    "everyone", "everything", "everywhere", "except", "few", "fifteen",
    "fifty", "fill", "find", "fire", "first", "five", "for", "former",
    "formerly", "forty", "found", "four", "from", "front", "full", "further",
    "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her",
    "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
    "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if",
    "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself",
    "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
    "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",
    "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
    "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no",
    "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of",
    "off", "often", "on", "once", "one", "only", "onto", "or", "other",
    "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own",
    "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
    "seem", "seemed", "seeming", "seems", "serious", "several", "she",
    "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
    "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
    "still", "such", "system", "take", "ten", "than", "that", "the", "their",
    "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves",
])

import config
from models import Chunk, DocStatus, RetrievalResult


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[RetrievalResult]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked:
            if scores[idx] > 0:
                results.append(
                    RetrievalResult(
                        chunk=self.chunks[idx],
                        score=float(scores[idx]),
                        retriever="bm25",
                    )
                )
        return results

    def save(self, path=None) -> None:
        """Save chunk data to SQLite. BM25 state is re-tokenized on load."""
        if path is None:
            path = config.BM25_INDEX_PATH
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("DROP TABLE IF EXISTS chunks")
            conn.execute("""
                CREATE TABLE chunks (
                    idx INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    doc_title TEXT NOT NULL,
                    doc_status TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL
                )
            """)

            conn.executemany(
                "INSERT INTO chunks (idx, chunk_id, text, source_file, doc_title, doc_status, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (i, c.chunk_id, c.text, c.source_file, c.doc_title, c.doc_status, c.chunk_index)
                    for i, c in enumerate(self.chunks)
                ],
            )
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def load(cls, path=None) -> "BM25Index":
        if path is None:
            path = config.BM25_INDEX_PATH

        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute(
                "SELECT chunk_id, text, source_file, doc_title, doc_status, chunk_index FROM chunks ORDER BY idx"
            ).fetchall()
        finally:
            conn.close()

        chunks = [
            Chunk(
                chunk_id=r[0], text=r[1], source_file=r[2],
                doc_title=r[3], doc_status=DocStatus(r[4]), chunk_index=r[5],
            )
            for r in rows
        ]

        bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

        instance = object.__new__(cls)
        instance.chunks = chunks
        instance.bm25 = bm25
        return instance
