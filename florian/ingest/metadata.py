import re

from models import DocStatus

_ARCHIVED_CONTENT_SIGNALS = [
    "archived document",
    "do not rely on this",
    "deprecated",
    "no longer maintained",
    "superseded by",
    "replaced by",
    "end of life",
    "eol",
]

# Pre-compiled word-boundary patterns for content signal matching.
# Prevents partial matches like "deprecated" inside "predeprecated".
_ARCHIVED_CONTENT_PATTERNS = [
    re.compile(r"\b" + re.escape(s) + r"\b") for s in _ARCHIVED_CONTENT_SIGNALS
]

_ARCHIVED_FILENAME_SIGNALS = [
    "archived",
    "historical",
    "legacy",
    "old",
    "deprecated",
]

_INFORMAL_CONTENT_SIGNALS = [
    "internal notes",
    "random notes",
    "do not treat this document as official",
    "may not represent final",
    "slack dump",
    "meeting notes",
    "brainstorm",
    "todo",
    "scratch",
]

_INFORMAL_CONTENT_PATTERNS = [
    re.compile(r"\b" + re.escape(s) + r"\b") for s in _INFORMAL_CONTENT_SIGNALS
]

_INFORMAL_PATH_SIGNALS = [
    "drafts",
    "scratch",
    "notes",
    "temp",
    "wip",
]

_ARCHIVED_PATH_SIGNALS = [
    "archive",
    "archived",
    "legacy",
    "old",
    "deprecated",
]

# Regex to detect YAML/TOML frontmatter status fields
_FRONTMATTER_STATUS_RE = re.compile(
    r"^---\s*\n.*?^status\s*:\s*(\w+)",
    re.MULTILINE | re.DOTALL,
)


def _filename_has_signal(filename: str, signal: str) -> bool:
    """Check if signal appears as a distinct component in the filename.

    Splits by common separators (_, -, .) so 'old' matches 'old_spec.md'
    but not 'bold_spec.md'.
    """
    parts = re.split(r"[_\-./\\]", filename)
    return signal in parts


def classify_document(content: str, filepath: str) -> str:
    """Return 'current', 'archived', or 'informal'.

    Checks (in priority order):
    1. Frontmatter 'status' field
    2. Content signals in first 30 lines (word-boundary matching)
    3. Filename signals (component matching)
    4. Directory path signals
    """
    # 1. Check frontmatter
    fm_match = _FRONTMATTER_STATUS_RE.search(content[:2000])
    if fm_match:
        status = fm_match.group(1).lower()
        if status in ("archived", "deprecated", "obsolete", "legacy"):
            return DocStatus.ARCHIVED
        if status in ("draft", "informal", "wip"):
            return DocStatus.INFORMAL

    first_lines = "\n".join(content.splitlines()[:30]).lower()
    filepath_lower = filepath.lower()
    filename_lower = filepath_lower.rsplit("/", 1)[-1] if "/" in filepath_lower else filepath_lower

    # 2. Content signals (word-boundary matching)
    for pattern in _ARCHIVED_CONTENT_PATTERNS:
        if pattern.search(first_lines):
            return DocStatus.ARCHIVED

    for pattern in _INFORMAL_CONTENT_PATTERNS:
        if pattern.search(first_lines):
            return DocStatus.INFORMAL

    # 3. Filename signals (component matching)
    for signal in _ARCHIVED_FILENAME_SIGNALS:
        if _filename_has_signal(filename_lower, signal):
            return DocStatus.ARCHIVED

    # 4. Directory path signals (only if file is in a subdirectory)
    if "/" in filepath_lower:
        dir_part = filepath_lower.rsplit("/", 1)[0]
        for signal in _ARCHIVED_PATH_SIGNALS:
            if signal in dir_part.split("/"):
                return DocStatus.ARCHIVED
        for signal in _INFORMAL_PATH_SIGNALS:
            if signal in dir_part.split("/"):
                return DocStatus.INFORMAL

    return DocStatus.CURRENT
