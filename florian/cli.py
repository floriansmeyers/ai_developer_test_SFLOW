import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

import config
from models import DocStatus
from pipeline import RAGPipeline

console = Console()


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Show retrieval details.")
@click.option("--reindex", is_flag=True, help="Force rebuild of the index.")
@click.pass_context
def cli(ctx, verbose, reindex):
    """Easify RAG — Ask questions about Easify platform documentation."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["reindex"] = reindex

    if ctx.invoked_subcommand is None:
        ctx.invoke(interactive)


@cli.command()
@click.argument("question")
@click.pass_context
def ask(ctx, question):
    """Ask a single question."""
    pipeline = RAGPipeline(verbose=ctx.obj["verbose"])
    pipeline.build_index(force=ctx.obj["reindex"])
    _ask_and_display(pipeline, question)


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start an interactive question-answer session."""
    pipeline = RAGPipeline(verbose=ctx.obj["verbose"])

    console.print(
        Panel(
            "[bold]Easify RAG System[/bold]\n"
            "Ask questions about the Easify platform documentation.\n"
            "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
            border_style="blue",
        )
    )

    with console.status("[bold green]Building index..."):
        pipeline.build_index(force=ctx.obj["reindex"])
    console.print("[green]Index ready.[/green]\n")

    while True:
        try:
            question = console.input("[bold blue]Question:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("Goodbye!")
            break

        _ask_and_display(pipeline, question)


@cli.command()
@click.option("--runs", default=1, type=int, help="Number of evaluation runs (>1 for stress test).")
@click.option("--faithfulness", is_flag=True, help="Run LLM-as-judge faithfulness checks (costs extra API calls).")
@click.pass_context
def evaluate(ctx, runs, faithfulness):
    """Run the evaluation suite against all test cases."""
    from evaluate.evaluator import run_assessment, run_stress_test

    pipeline = RAGPipeline(verbose=ctx.obj["verbose"])
    pipeline.build_index(force=ctx.obj["reindex"])

    if runs > 1:
        run_stress_test(pipeline, runs=runs)
    else:
        run_assessment(pipeline, check_faithfulness=faithfulness)


def _ask_and_display(pipeline: RAGPipeline, question: str) -> None:
    """Run the pipeline and display a formatted answer."""
    console.print()
    with console.status("[bold green]Thinking..."):
        answer, results = pipeline.answer(question)

    console.print(Markdown(answer))
    console.print()

    # Show sources
    sources = list({r.chunk.source_file: r for r in results}.values())
    if sources:
        source_lines = []
        for r in sources:
            status = f" \\[{r.chunk.doc_status.value}]" if r.chunk.doc_status != DocStatus.CURRENT else ""
            source_lines.append(f"  - {r.chunk.source_file}{status}")
        console.print(
            Panel(
                "\n".join(source_lines),
                title="Retrieved Sources",
                border_style="dim",
            )
        )
    console.print()


if __name__ == "__main__":
    cli()
