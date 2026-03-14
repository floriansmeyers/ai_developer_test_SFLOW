from models import Chunk, RetrievalResult
from generate.prompt import build_user_prompt, SYSTEM_PROMPT


def _make_result(chunk_id: str, score: float) -> RetrievalResult:
    chunk = Chunk(
        text=f"Content of {chunk_id}",
        chunk_id=chunk_id,
        source_file=f"{chunk_id}.md",
        doc_title=chunk_id,
        doc_status="current",
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=score, retriever="test")


def test_build_user_prompt_contains_question():
    results = [_make_result("doc1", 0.9)]
    prompt = build_user_prompt("What is a unit?", results)
    assert "What is a unit?" in prompt
    assert "Context documents:" in prompt
    assert "doc1.md" in prompt


def test_build_user_prompt_includes_status():
    chunk = Chunk(
        text="Old info",
        chunk_id="old_0",
        source_file="old.md",
        doc_title="Old",
        doc_status="archived",
        chunk_index=0,
    )
    result = RetrievalResult(chunk=chunk, score=0.5, retriever="test")
    prompt = build_user_prompt("Question?", [result])
    assert "ARCHIVED" in prompt


def test_system_prompt_has_injection_guard():
    assert "override" in SYSTEM_PROMPT.lower() or "ignore" in SYSTEM_PROMPT.lower()


def test_system_prompt_has_uncertainty_rule():
    assert "unsure" in SYSTEM_PROMPT.lower()


def test_system_prompt_has_contradiction_rule():
    assert "contradict" in SYSTEM_PROMPT.lower()
