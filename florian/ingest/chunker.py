import re

from models import Document, Chunk

# Hard cap: chunks exceeding this word count get recursively split
CHUNK_HARD_CAP_WORDS = 300

# Overlap: number of words from the end of the previous chunk prepended to the next
CHUNK_OVERLAP_WORDS = 40


def _split_by_sections(content: str) -> list[str]:
    """Split on markdown headers (## or #) or double newlines."""
    sections = re.split(r"\n(?=#{1,2}\s)", content)
    if len(sections) > 1:
        return [s.strip() for s in sections if s.strip()]

    # Fall back to double-newline splitting
    paragraphs = re.split(r"\n\s*\n", content)
    # Merge very small paragraphs back together
    merged = []
    current = ""
    for p in paragraphs:
        if len((current + " " + p).split()) < 120:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                merged.append(current)
            current = p
    if current:
        merged.append(current)
    return merged if merged else [content]


def _split_oversized(text: str, max_words: int) -> list[str]:
    """Recursively split text that exceeds max_words by sentences, then by words."""
    if len(text.split()) <= max_words:
        return [text]

    # Try splitting by sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        parts = []
        current = ""
        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate.split()) <= max_words:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = sent
        if current:
            parts.append(current)
        # Recursively split any still-oversized parts
        result = []
        for part in parts:
            result.extend(_split_oversized(part, max_words))
        return result

    # Last resort: split by words
    words = text.split()
    parts = []
    for i in range(0, len(words), max_words):
        parts.append(" ".join(words[i : i + max_words]))
    return parts


def _add_overlap(sections: list[str], overlap_words: int) -> list[str]:
    """Prepend the last N words of the previous section to each subsequent section."""
    if overlap_words <= 0 or len(sections) <= 1:
        return sections

    result = [sections[0]]
    for i in range(1, len(sections)):
        prev_words = sections[i - 1].split()
        overlap_text = " ".join(prev_words[-overlap_words:])
        result.append(f"...{overlap_text}\n\n{sections[i]}")
    return result


def chunk_documents(documents: list[Document], max_words: int = 120) -> list[Chunk]:
    """Convert documents into chunks. Small docs stay whole; large ones get split."""
    chunks = []
    for doc in documents:
        word_count = len(doc.content.split())
        if word_count <= max_words:
            sections = [doc.content]
        else:
            sections = _split_by_sections(doc.content)

        # Hard cap: split any oversized sections
        capped = []
        for section in sections:
            capped.extend(_split_oversized(section, CHUNK_HARD_CAP_WORDS))
        sections = capped

        # Add overlap between consecutive chunks
        sections = _add_overlap(sections, CHUNK_OVERLAP_WORDS)

        chunk_idx = 0
        for section_text in sections:
            # Prepend doc title as context when splitting
            if len(sections) > 1:
                text = f"[{doc.doc_title}]\n\n{section_text}"
            else:
                text = section_text

            # Final hard cap enforcement after title/overlap additions
            final_texts = _split_oversized(text, CHUNK_HARD_CAP_WORDS)

            safe_id = doc.source_file.replace("/", "__").replace("\\", "__")
            for final_text in final_texts:
                chunks.append(
                    Chunk(
                        text=final_text,
                        chunk_id=f"{safe_id}_{chunk_idx}",
                        source_file=doc.source_file,
                        doc_title=doc.doc_title,
                        doc_status=doc.doc_status,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1
    return chunks
