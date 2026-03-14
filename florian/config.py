import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT.parent / "files"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
BM25_INDEX_PATH = PROJECT_ROOT / "bm25_index.sqlite"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.1

_openai_client: OpenAI | None = None


def validate_api_key() -> None:
    if not OPENAI_API_KEY:
        raise SystemExit(
            "Error: OPENAI_API_KEY is not set.\n"
            "Create a .env file with your key: OPENAI_API_KEY=sk-..."
        )


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        validate_api_key()
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

from models import DocStatus

# Chunking
CHUNK_MAX_WORDS = 120

# Retrieval
RETRIEVAL_TOP_K = 10
RRF_K = 60
RERANK_TOP_N = 5

# Metadata soft weights
METADATA_WEIGHTS = {
    DocStatus.CURRENT: 1.0,
    DocStatus.ARCHIVED: 0.6,
    DocStatus.INFORMAL: 0.7,
}

# Cross-encoder - Best to leave enabled as it provides a significant boost in answer quality.
USE_CROSS_ENCODER = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CONFIDENCE_THRESHOLD = -3.0  # ms-marco score; below this, refuse without LLM call

LLM_MAX_TOKENS = 1024

# Retry
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0

_logger = logging.getLogger(__name__)


def retry_api_call(fn):
    """Exponential backoff on transient API errors (429, 5xx, timeouts)."""
    for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            is_retryable = status in (429, 500, 502, 503) or "timeout" in str(exc).lower()
            if not is_retryable or attempt == _RETRY_MAX_ATTEMPTS:
                raise
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            _logger.warning("API call failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt, _RETRY_MAX_ATTEMPTS, exc, delay)
            time.sleep(delay)
