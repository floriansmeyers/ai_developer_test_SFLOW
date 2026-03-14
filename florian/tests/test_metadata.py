"""Tests for ingest/metadata.py."""

from ingest.metadata import classify_document


def test_archived_by_content():
    content = "ARCHIVED DOCUMENT\nThis spec is no longer valid."
    assert classify_document(content, "some_file.md") == "archived"


def test_archived_by_deprecated_signal():
    content = "This feature is deprecated and should not be used."
    assert classify_document(content, "old_spec.md") == "archived"


def test_archived_by_filename():
    content = "Normal document content."
    assert classify_document(content, "archived_specs_2019.md") == "archived"


def test_archived_by_historical_filename():
    content = "Normal document content."
    assert classify_document(content, "historical_changes.md") == "archived"


def test_informal_by_content():
    content = "Internal notes from engineering meeting."
    assert classify_document(content, "notes.txt") == "informal"


def test_informal_slack_dump():
    content = "This is a slack dump from the team channel."
    assert classify_document(content, "dump.txt") == "informal"


def test_current_by_default():
    content = "This document describes the project model hierarchy."
    assert classify_document(content, "project_model.md") == "current"


def test_scan_range_covers_later_lines():
    """Signals on line 20 should still be detected (scan range is 30 lines)."""
    lines = ["Normal line."] * 19 + ["This is deprecated and replaced."]
    content = "\n".join(lines)
    assert classify_document(content, "spec.md") == "archived"


def test_word_boundary_content_no_false_positive():
    """'deprecated' inside 'predeprecated' should NOT trigger archived status."""
    content = "This system is predeprecated but still active."
    assert classify_document(content, "spec.md") == "current"


def test_word_boundary_content_matches_exact():
    """'deprecated' as a standalone word should trigger archived status."""
    content = "This feature is deprecated."
    assert classify_document(content, "spec.md") == "archived"


def test_filename_component_matching_no_false_positive():
    """'old' inside 'bold' should NOT trigger archived status."""
    content = "Normal document content."
    assert classify_document(content, "bold_design.md") == "current"


def test_filename_component_matching_exact():
    """'old' as a filename component should trigger archived status."""
    content = "Normal document content."
    assert classify_document(content, "old_design.md") == "archived"
