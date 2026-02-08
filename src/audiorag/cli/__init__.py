"""Premium CLI for AudioRAG."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.theme import Theme

from audiorag import AudioRAGConfig, AudioRAGPipeline

# Top-tier design theme: Minimalist, sophisticated colors, no emojis
AUDIORAG_THEME = Theme(
    {
        "info": "bold cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "highlight": "bold magenta",
        "dim": "grey50",
    }
)

# Custom style for questionary to match our premium theme and remove all background highlights
QUESTIONARY_STYLE = questionary.Style(
    [
        ("qmark", "fg:#00ffff bold"),  # Cyan question mark
        ("question", "bold"),
        ("answer", "fg:#00ff00 bold"),  # Green answer
        ("pointer", "fg:#00ffff bold"),  # Cyan pointer
        ("highlighted", "fg:#00ffff bold bg:default noreverse"),  # Cyan text for current item
        ("selected", "fg:default bg:default noreverse"),  # Default choice style (no box)
        ("choice", "fg:default bg:default noreverse"),  # Standard choice style
    ]
)

console = Console(theme=AUDIORAG_THEME)


def validate_api_key(provider: str):
    """Return a validation function for a specific provider."""

    def validator(text: str) -> bool | str:
        if not text:
            return f"{provider.upper()} API key cannot be empty"

        p = provider.lower()
        if p == "openai":
            if not text.startswith("sk-"):
                return "OpenAI keys must start with 'sk-'"
            if len(text) < 40:
                return "OpenAI key is too short"
        elif p == "anthropic":
            if not text.startswith("sk-ant-"):
                return "Anthropic keys must start with 'sk-ant-'"
        elif p == "groq":
            if not text.startswith("gsk_"):
                return "Groq keys must start with 'gsk_'"
        elif p == "voyage":
            if not text.startswith("pa-"):
                return "Voyage keys must start with 'pa-'"
        elif p == "deepgram":
            if len(text) != 40:
                return "Deepgram keys are usually 40 characters"
        elif p == "assemblyai":
            if len(text) != 32:
                return "AssemblyAI keys are usually 32 characters"

        if len(text) < 20:
            return f"{provider.upper()} key seems too short"

        return True

    return validator


async def setup_cmd() -> None:
    """Run interactive setup to create .env file using arrow-key selection."""
    console.print(Panel("AudioRAG Configuration Setup", style="info", expand=False))
    console.print("This utility will generate a .env file for your providers.\n", style="dim")

    try:
        env_lines = ["# AudioRAG Configuration\n"]

        # 1. STT
        stt_provider = await questionary.select(
            "Select Speech-to-Text (STT) Provider:",
            choices=["openai", "deepgram", "groq", "assemblyai"],
            style=QUESTIONARY_STYLE,
        ).ask_async()
        if stt_provider is None:
            return
        env_lines.append(f"AUDIORAG_STT_PROVIDER={stt_provider}\n")

        stt_key = await questionary.password(
            f"Enter {stt_provider.upper()} API Key:",
            validate=validate_api_key(stt_provider),
            style=QUESTIONARY_STYLE,
        ).ask_async()
        if stt_key is None:
            return
        key_var = f"AUDIORAG_{stt_provider.upper()}_API_KEY"
        env_lines.append(f"{key_var}={stt_key}\n")

        # 2. Embedding
        embedding_provider = await questionary.select(
            "Select Embedding Provider:",
            choices=["openai", "voyage", "cohere"],
            style=QUESTIONARY_STYLE,
        ).ask_async()
        if embedding_provider is None:
            return
        env_lines.append(f"AUDIORAG_EMBEDDING_PROVIDER={embedding_provider}\n")

        embed_key = await questionary.password(
            f"Enter {embedding_provider.upper()} API Key:",
            validate=validate_api_key(embedding_provider),
            style=QUESTIONARY_STYLE,
        ).ask_async()
        if embed_key is None:
            return
        key_var = f"AUDIORAG_{embedding_provider.upper()}_API_KEY"
        env_lines.append(f"{key_var}={embed_key}\n")

        # 3. Vector Store
        vector_store = await questionary.select(
            "Select Vector Store:",
            choices=["chromadb", "pinecone", "weaviate", "supabase"],
            style=QUESTIONARY_STYLE,
        ).ask_async()
        if vector_store is None:
            return
        env_lines.append(f"AUDIORAG_VECTOR_STORE_PROVIDER={vector_store}\n")

        if vector_store != "chromadb":
            if vector_store == "supabase":
                conn_str = await questionary.text(
                    "Enter Supabase Connection String:",
                    validate=lambda t: True if t.startswith("postgresql://") else "Invalid format",
                    style=QUESTIONARY_STYLE,
                ).ask_async()
                if conn_str:
                    env_lines.append(f"AUDIORAG_SUPABASE_CONNECTION_STRING={conn_str}\n")
            else:
                vs_key = await questionary.password(
                    f"Enter {vector_store.upper()} API Key:",
                    validate=validate_api_key(vector_store),
                    style=QUESTIONARY_STYLE,
                ).ask_async()
                if vs_key:
                    env_lines.append(f"AUDIORAG_{vector_store.upper()}_API_KEY={vs_key}\n")

        # 4. Generation
        gen_provider = await questionary.select(
            "Select Generation Provider:",
            choices=["openai", "anthropic", "gemini"],
            style=QUESTIONARY_STYLE,
        ).ask_async()
        if gen_provider is None:
            return
        env_lines.append(f"AUDIORAG_GENERATION_PROVIDER={gen_provider}\n")

        gen_key = await questionary.password(
            f"Enter {gen_provider.upper()} API Key:",
            validate=validate_api_key(gen_provider),
            style=QUESTIONARY_STYLE,
        ).ask_async()
        if gen_key is None:
            return
        key_var = (
            "AUDIORAG_GOOGLE_API_KEY"
            if gen_provider == "gemini"
            else f"AUDIORAG_{gen_provider.upper()}_API_KEY"
        )
        env_lines.append(f"{key_var}={gen_key}\n")

        new_config = {}
        for line in env_lines:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                new_config[key] = val

        env_path = Path(".env")
        final_lines = []

        if env_path.exists():
            with open(env_path, "r") as f:
                existing_lines = f.readlines()

            for line in existing_lines:
                if (
                    not line.strip()
                    or line.strip().startswith("#")
                    or not line.strip().startswith("AUDIORAG_")
                ):
                    final_lines.append(line)

            if final_lines and not final_lines[-1].endswith("\n"):
                final_lines.append("\n")
            final_lines.append("\n# Updated AudioRAG Configuration\n")
        else:
            final_lines.append("# AudioRAG Configuration\n")

        for key, val in new_config.items():
            final_lines.append(f"{key}={val}\n")

        with open(env_path, "w") as f:
            f.writelines(final_lines)

        console.print("\n[success]Configuration updated in .env[/]")

    except KeyboardInterrupt:
        return


async def index_cmd(url: str, force: bool) -> None:
    """Index audio from a URL with premium progress tracking."""
    config = AudioRAGConfig()
    pipeline = AudioRAGPipeline(config)

    console.print(f"[highlight]Starting pipeline for:[/] {url}\n")

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[info]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[dim]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description="Processing", total=100)

        try:
            await pipeline.index(url, force=force)
            progress.update(task, completed=100, description="Done")
            console.print(f"[success]Indexing complete:[/] {url}")
        except Exception as e:
            console.print(f"[error]Indexing failed:[/] {e}")
            sys.exit(1)


async def query_cmd(query_text: str) -> None:
    """Perform a RAG query and display results in a top-tier layout."""
    config = AudioRAGConfig()
    pipeline = AudioRAGPipeline(config)

    with console.status("[info]Searching index...", spinner="dots"):
        try:
            result = await pipeline.query(query_text)

            console.print("\n")
            console.print(
                Panel(
                    result.answer,
                    title="AudioRAG Answer",
                    title_align="left",
                    border_style="success",
                    padding=(1, 2),
                )
            )

            if result.sources:
                table = Table(
                    box=None,
                    show_header=True,
                    header_style="highlight",
                    title="Supporting Sources",
                    title_justify="left",
                    title_style="dim",
                    pad_edge=False,
                )
                table.add_column("#", style="dim", width=2)
                table.add_column("Source", ratio=3)
                table.add_column("Timestamp", justify="right", ratio=1)
                table.add_column("Relevance", justify="right", style="success", ratio=1)

                for i, source in enumerate(result.sources, 1):
                    table.add_row(
                        str(i),
                        source.title,
                        f"{source.start_time:.1f}s - {source.end_time:.1f}s",
                        f"{source.relevance_score:.2f}",
                    )
                console.print(table)
                console.print("\n")
        except Exception as e:
            console.print(f"[error]Query failed:[/] {e}")
            sys.exit(1)


def main() -> None:
    """Entry point with clean help documentation."""
    parser = argparse.ArgumentParser(
        description="AudioRAG: Professional Audio RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  audiorag setup
  audiorag index "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  audiorag query "What is the key takeaway?"
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Setup
    subparsers.add_parser("setup", help="Initialize provider configuration")

    # Index
    index_parser = subparsers.add_parser("index", help="Process and index audio from a URL")
    index_parser.add_argument("url", help="URL of the audio/video to index")
    index_parser.add_argument(
        "--force", action="store_true", help="Re-process even if already indexed"
    )

    # Query
    query_parser = subparsers.add_parser("query", help="Query the semantic index")
    query_parser.add_argument("text", help="The question to ask the audio index")

    args = parser.parse_args()

    try:
        if args.command == "setup":
            asyncio.run(setup_cmd())
        elif args.command == "index":
            asyncio.run(index_cmd(args.url, args.force))
        elif args.command == "query":
            asyncio.run(query_cmd(args.text))
        else:
            parser.print_help()
    except KeyboardInterrupt:
        console.print("\n[warning]Operation cancelled by user.[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()
