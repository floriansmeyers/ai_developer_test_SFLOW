import tempfile
from pathlib import Path

from ingest.loader import load_documents, _is_supported


def test_recursive_traversal():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "top.md").write_text("Top-level doc.")
        sub = root / "subdir"
        sub.mkdir()
        (sub / "nested.md").write_text("Nested doc.")

        docs = load_documents(root)
        sources = {d.source_file for d in docs}
        assert "top.md" in sources
        assert "subdir/nested.md" in sources


def test_hidden_files_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / ".hidden.md").write_text("Hidden file.")
        (root / "visible.md").write_text("Visible file.")

        docs = load_documents(root)
        assert len(docs) == 1
        assert docs[0].source_file == "visible.md"


def test_unsupported_extension_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (root / "doc.md").write_text("Real doc.")

        docs = load_documents(root)
        assert len(docs) == 1
        assert docs[0].source_file == "doc.md"


def test_empty_file_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "empty.md").write_text("")
        (root / "content.md").write_text("Has content.")

        docs = load_documents(root)
        assert len(docs) == 1


def test_supported_extensions():
    assert _is_supported(Path("file.md"))
    assert _is_supported(Path("file.txt"))
    assert not _is_supported(Path("file.pdf"))
    assert not _is_supported(Path("file.docx"))
    assert not _is_supported(Path("file.png"))
    assert not _is_supported(Path("file.zip"))
    assert not _is_supported(Path("file.exe"))


def test_subdirectory_source_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        deep = root / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "deep.md").write_text("Deep nested doc.")

        docs = load_documents(root)
        assert docs[0].source_file == "a/b/deep.md"
