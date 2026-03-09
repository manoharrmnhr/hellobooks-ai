"""Hellobooks AI — Command Line Interface"""
from __future__ import annotations
import logging
import typer
import uvicorn
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from src.config import settings
from src.rag_engine import RAGEngine

app = typer.Typer(name="hellobooks", help="Hellobooks AI — RAG-powered accounting assistant", add_completion=False)
console = Console()
_engine = None

def _get_engine(auto_ingest=True):
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        if auto_ingest:
            with console.status("[bold green]Loading knowledge base…"):
                _engine.ingest()
    return _engine

@app.command()
def ingest(force: bool = typer.Option(False, "--force", "-f", help="Force rebuild")):
    """Build or rebuild the FAISS index from knowledge base documents."""
    logging.basicConfig(level=logging.WARNING)
    engine = RAGEngine()
    with console.status("[bold green]Ingesting knowledge base…"):
        n = engine.ingest(force_rebuild=force)
    console.print(Panel(
        f"[green]✓ Index built successfully[/green]\n"
        f"[dim]Chunks indexed:[/dim] [bold]{n}[/bold]\n"
        f"[dim]Embedding model:[/dim] [bold]{settings.embedding_model}[/bold]\n"
        f"[dim]Index saved to:[/dim] [bold]{settings.faiss_index_path}[/bold]",
        title="[bold]Hellobooks AI — Ingestion Complete[/bold]", border_style="green"))

@app.command()
def query(
    question: str = typer.Argument(..., help="The accounting question to ask"),
    top_k: int = typer.Option(3, "--top-k", "-k"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ask a single accounting question."""
    logging.basicConfig(level=logging.WARNING)
    engine = _get_engine()
    with console.status("[bold cyan]Thinking…"):
        response = engine.query(question, top_k=top_k)
    console.print(Panel(Markdown(response.answer), title="[bold cyan]Answer[/bold cyan]", border_style="cyan"))
    console.print(f"\n[dim]Sources:[/dim] {' · '.join(response.sources)}")
    console.print(f"[dim]Latency:[/dim] {response.latency_ms:.0f}ms  [dim]Model:[/dim] {response.model_used}")
    if verbose:
        table = Table(title="Retrieved Chunks", border_style="dim")
        table.add_column("Rank", width=5)
        table.add_column("Source", width=22)
        table.add_column("Score", width=7)
        table.add_column("Preview")
        for i, chunk in enumerate(response.retrieved_chunks, 1):
            table.add_row(str(i), chunk.source_label, f"{chunk.score:.3f}", chunk.document.content[:100] + "…")
        console.print(table)

@app.command()
def chat():
    """Start an interactive chat REPL session."""
    logging.basicConfig(level=logging.WARNING)
    console.print(Panel(
        "[bold green]Welcome to Hellobooks AI[/bold green]\n"
        "Ask any accounting question. Type [bold]exit[/bold] to leave.\n"
        "Commands: [bold]/verbose[/bold] toggle · [bold]/help[/bold]",
        title="🏦 Hellobooks AI Chat", border_style="green"))
    engine = _get_engine()
    verbose = False
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]"); break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]"); break
        if user_input == "/verbose":
            verbose = not verbose
            console.print(f"[dim]Verbose: {'on' if verbose else 'off'}[/dim]"); continue
        if user_input == "/help":
            console.print("[dim]/verbose — toggle chunks · exit — quit[/dim]"); continue
        with console.status("[bold cyan]Thinking…"):
            response = engine.query(user_input)
        console.print(f"\n[bold green]Hellobooks AI[/bold green]")
        console.print(Markdown(response.answer))
        console.print(f"\n[dim]Sources: {' · '.join(response.sources)} | {response.latency_ms:.0f}ms[/dim]")
        if verbose:
            for i, chunk in enumerate(response.retrieved_chunks, 1):
                console.print(f"  [dim][{i}] {chunk.source_label} (score={chunk.score:.3f}): {chunk.document.content[:80]}…[/dim]")

@app.command()
def serve(
    host: str = typer.Option(settings.api_host, "--host", "-H"),
    port: int = typer.Option(settings.api_port, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r"),
):
    """Launch the FastAPI HTTP server."""
    console.print(Panel(
        f"[bold green]Starting Hellobooks AI API[/bold green]\n"
        f"[dim]Host:[/dim] {host}  [dim]Port:[/dim] {port}\n"
        f"[dim]Docs:[/dim] http://{host}:{port}/docs",
        title="🚀 API Server", border_style="green"))
    uvicorn.run("src.api:app", host=host, port=port, reload=reload, log_level=settings.log_level.lower())

if __name__ == "__main__":
    app()
