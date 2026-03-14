from dataclasses import dataclass
from enum import Enum


class DocStatus(str, Enum):
    CURRENT = "current"
    ARCHIVED = "archived"
    INFORMAL = "informal"


@dataclass
class Document:
    content: str
    source_file: str
    doc_title: str
    doc_status: DocStatus


@dataclass
class Chunk:
    text: str
    chunk_id: str  # e.g. "finance_notes.md_0"
    source_file: str
    doc_title: str
    doc_status: DocStatus
    chunk_index: int


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    retriever: str  # "vector" | "bm25" | "hybrid" | "reranked"
