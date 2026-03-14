"""Run the assessment suite and report results."""

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from evaluate.test_cases import TEST_CASES

console = Console()


def _check_retrieval(result_sources: list[str], expected_sources: list[str]) -> tuple[bool, str]:
    """Check if at least one expected source was retrieved.

    Only checks against the top RERANK_TOP_N sources (those that reach the LLM).
    """
    import config

    if not expected_sources:
        return True, "N/A (refusal test)"

    result_sources = result_sources[:config.RERANK_TOP_N]
    found = [s for s in expected_sources if s in result_sources]
    if found:
        return True, f"Found: {', '.join(found)}"
    return False, f"Missing: {', '.join(expected_sources)}"


def _check_answer_content(answer: str, test_case: dict) -> tuple[bool, str]:
    """Check required and forbidden phrases in the answer."""
    answer_lower = answer.lower()
    issues = []

    required = test_case.get("required_in_answer", [])
    require_any = test_case.get("required_any", False)

    if required:
        if require_any:
            if not any(phrase.lower() in answer_lower for phrase in required):
                issues.append(f"None of required phrases found: {required}")
        else:
            for phrase in required:
                if phrase.lower() not in answer_lower:
                    issues.append(f"Missing required: '{phrase}'")

    for phrase in test_case.get("forbidden_in_answer", []):
        if phrase.lower() in answer_lower:
            issues.append(f"Contains forbidden: '{phrase}'")

    if issues:
        return False, "; ".join(issues)
    return True, "OK"


def _check_faithfulness(question: str, answer: str, context_texts: list[str]) -> tuple[bool, str]:
    """Use LLM-as-judge to check if the answer is faithful to the retrieved context.

    Returns (passed, detail). Failures here are warnings, not hard failures,
    since LLM-as-judge is itself non-deterministic.
    """
    import config

    if not config.OPENAI_API_KEY or not context_texts:
        return True, "skipped"

    context_block = "\n---\n".join(context_texts[:5])

    judge_prompt = (
        "You are an assessment judge. Determine if the ANSWER is faithful to the CONTEXT — "
        "meaning every claim in the answer is supported by or directly inferable from the context.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        "Respond with ONLY one word: FAITHFUL or UNFAITHFUL"
    )

    try:
        client = config.get_openai_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0.0,
            max_tokens=10,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        verdict = response.choices[0].message.content.strip().upper()
        if "FAITHFUL" in verdict and "UNFAITHFUL" not in verdict:
            return True, "faithful"
        return False, "unfaithful"
    except Exception as e:
        return True, f"judge error: {e}"


def _evaluate_once(pipeline, check_faithfulness: bool = False) -> list[dict]:
    """Run all test cases once and return per-question results."""
    results = []
    for tc in TEST_CASES:
        answer, search_results = pipeline.answer(tc["question"])
        result_sources = [r.chunk.source_file for r in search_results]

        retrieval_ok, retrieval_detail = _check_retrieval(result_sources, tc["expected_sources"])
        answer_ok, answer_detail = _check_answer_content(answer, tc)

        faith_ok, faith_detail = True, "skipped"
        if check_faithfulness and tc["category"] != "refusal":
            context_texts = [r.chunk.text for r in search_results]
            faith_ok, faith_detail = _check_faithfulness(tc["question"], answer, context_texts)

        all_ok = retrieval_ok and answer_ok

        results.append({
            "question": tc["question"],
            "category": tc["category"],
            "retrieval_ok": retrieval_ok,
            "retrieval_detail": retrieval_detail,
            "answer_ok": answer_ok,
            "answer_detail": answer_detail,
            "faithfulness_ok": faith_ok,
            "faithfulness_detail": faith_detail,
            "passed": all_ok,
        })
    return results


def run_assessment(pipeline, check_faithfulness: bool = False) -> None:
    """Run all test cases and display results."""
    table = Table(title="Assessment Results", show_lines=True)
    table.add_column("Question", style="bold", max_width=45)
    table.add_column("Category", style="dim")
    table.add_column("Retrieval", max_width=30)
    table.add_column("Answer", max_width=30)
    if check_faithfulness:
        table.add_column("Faithfulness", max_width=15)
    table.add_column("Status")

    results = _evaluate_once(pipeline, check_faithfulness=check_faithfulness)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for r in results:
        retrieval_style = "[green]" if r["retrieval_ok"] else "[red]"
        answer_style = "[green]" if r["answer_ok"] else "[red]"
        status = "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]"

        row = [
            r["question"][:45],
            r["category"],
            f"{retrieval_style}{r['retrieval_detail']}[/]",
            f"{answer_style}{r['answer_detail']}[/]",
        ]
        if check_faithfulness:
            faith_style = "[green]" if r["faithfulness_ok"] else "[yellow]"
            row.append(f"{faith_style}{r['faithfulness_detail']}[/]")
        row.append(status)

        table.add_row(*row)

    console.print()
    console.print(table)
    console.print(f"\n[bold]Result: {passed}/{total} tests passed.[/bold]")

    if check_faithfulness:
        faith_pass = sum(1 for r in results if r["faithfulness_ok"])
        console.print(f"[bold]Faithfulness: {faith_pass}/{total} passed LLM-as-judge check.[/bold]")


def run_stress_test(pipeline, runs: int = 10) -> None:
    """Run the assessment N times and display aggregate statistics."""
    total_questions = len(TEST_CASES)
    question_categories = {tc["question"]: tc["category"] for tc in TEST_CASES}
    question_pass_counts: dict[str, int] = defaultdict(int)
    run_scores: list[int] = []

    for run_idx in range(1, runs + 1):
        console.print(f"\n[bold cyan]--- Run {run_idx}/{runs} ---[/bold cyan]")
        results = _evaluate_once(pipeline)
        passed = 0
        for r in results:
            if r["passed"]:
                question_pass_counts[r["question"]] += 1
                passed += 1
        run_scores.append(passed)
        console.print(f"  Score: {passed}/{total_questions}")

    pass_rates = {q: question_pass_counts.get(q, 0) / runs for q in question_categories}

    # Per-question summary table
    summary = Table(title=f"Stress Test Summary ({runs} runs)", show_lines=True)
    summary.add_column("Question", style="bold", max_width=50)
    summary.add_column("Category", style="dim")
    summary.add_column("Pass Rate", justify="right")

    for q, cat in question_categories.items():
        rate = pass_rates[q]
        if rate == 1.0:
            style = "[green]"
        elif rate >= 0.7:
            style = "[yellow]"
        else:
            style = "[red]"
        summary.add_row(q[:50], cat, f"{style}{question_pass_counts.get(q, 0)}/{runs}[/]")

    console.print()
    console.print(summary)

    # Per-category averages
    cat_totals: dict[str, list[float]] = defaultdict(list)
    for q, cat in question_categories.items():
        cat_totals[cat].append(pass_rates[q])

    cat_table = Table(title="Per-Category Average Pass Rate", show_lines=True)
    cat_table.add_column("Category", style="bold")
    cat_table.add_column("Avg Pass Rate", justify="right")

    for cat, rates in sorted(cat_totals.items()):
        avg = sum(rates) / len(rates)
        style = "[green]" if avg >= 0.9 else "[yellow]" if avg >= 0.7 else "[red]"
        cat_table.add_row(cat, f"{style}{avg:.0%}[/]")

    console.print()
    console.print(cat_table)

    # Aggregate stats
    mean_score = sum(run_scores) / len(run_scores)
    min_score = min(run_scores)
    max_score = max(run_scores)

    console.print()
    console.print(f"[bold]Aggregate across {runs} runs:[/bold]")
    console.print(f"  Mean: {mean_score:.1f}/{total_questions}")
    console.print(f"  Min:  {min_score}/{total_questions}")
    console.print(f"  Max:  {max_score}/{total_questions}")
