import logging
from pathlib import Path

from models import Document
from ingest.metadata import classify_document

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = {".md", ".txt"}


def _extract_title(content: str, filename: str) -> str:
    for line in content.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped and not stripped.startswith("ARCHIVED"):
            return stripped
    return filename


def _is_supported(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == "":
        return True  # extensionless files treated as plain text
    return suffix in _TEXT_EXTENSIONS


def load_documents(docs_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        if not _is_supported(path):
            logger.debug("Skipping unsupported file type: %s", path)
            continue

        relative_path = str(path.relative_to(docs_dir))
        content = path.read_text(encoding="utf-8", errors="replace")

        if not content.strip():
            logger.debug("Skipping empty file: %s", path)
            continue

        doc_status = classify_document(content, relative_path)
        doc_title = _extract_title(content, path.name)
        documents.append(
            Document(
                content=content,
                source_file=relative_path,
                doc_title=doc_title,
                doc_status=doc_status,
            )
        )
    return documents
