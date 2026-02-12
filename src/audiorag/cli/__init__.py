"""Premium CLI for AudioRAG."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from audiorag import AudioRAGConfig, AudioRAGPipeline

_active_pipeline: AudioRAGPipeline | None = None  # Set during index/query for signal handling

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


def validate_api_key(provider: str) -> Callable[[str], bool | str]:
    """Return a validation function for a specific provider."""
    prefixes = {
        "openai": "sk-",
        "anthropic": "sk-ant-",
        "groq": "gsk_",
        "voyage": "pa-",
    }
    lengths = {
        "openai": 40,
        "deepgram": 40,
        "assemblyai": 32,
    }

    def validator(text: str) -> bool | str:
        if not text:
            return f"{provider.upper()} API key cannot be empty"

        p = provider.lower()
        prefix = prefixes.get(p)
        if prefix and not text.startswith(prefix):
            return f"{provider.upper()} keys must start with '{prefix}'"

        min_len = lengths.get(p, 20)
        if len(text) < min_len:
            return f"{provider.upper()} key is too short"

        return True

    return validator


async def _get_provider_config(category: str, choices: list[str], env_key: str) -> list[str] | None:
    """Helper to prompt for a provider and its corresponding API key."""
    provider = await questionary.select(
        f"Select {category} Provider:",
        choices=choices,
        style=QUESTIONARY_STYLE,
    ).ask_async()
    if provider is None:
        return None

    lines = [f"AUDIORAG_{env_key}_PROVIDER={provider}\n"]

    if env_key == "VECTOR_STORE":
        if provider == "chromadb":
            return lines
        if provider == "supabase":
            conn_str = await questionary.text(
                "Enter Supabase Connection String:",
                validate=lambda t: True if t.startswith("postgresql://") else "Invalid format",
                style=QUESTIONARY_STYLE,
            ).ask_async()
            if conn_str is None:
                return None
            if conn_str:
                lines.append(f"AUDIORAG_SUPABASE_CONNECTION_STRING={conn_str}\n")
            return lines

    key = await questionary.password(
        f"Enter {provider.upper()} API Key:",
        validate=validate_api_key(provider),
        style=QUESTIONARY_STYLE,
    ).ask_async()

    if key is None:
        return None

    key_var = (
        "AUDIORAG_GOOGLE_API_KEY"
        if provider == "gemini"
        else f"AUDIORAG_{provider.upper()}_API_KEY"
    )
    lines.append(f"{key_var}={key}\n")
    return lines


async def setup_cmd() -> None:
    """Run interactive setup to create .env file using arrow-key selection."""
    console.print(Panel("AudioRAG Configuration Setup", style="info", expand=False))
    console.print("This utility will generate a .env file for your providers.\n", style="dim")

    try:
        env_lines = ["# AudioRAG Configuration\n"]

        configs = [
            ("Speech-to-Text (STT)", ["openai", "deepgram", "groq", "assemblyai"], "STT"),
            ("Embedding", ["openai", "voyage", "cohere"], "EMBEDDING"),
            ("Vector Store", ["chromadb", "pinecone", "weaviate", "supabase"], "VECTOR_STORE"),
            ("Generation", ["openai", "anthropic", "gemini"], "GENERATION"),
        ]

        for category, choices, env_key in configs:
            res = await _get_provider_config(category, choices, env_key)
            if res is None:
                return
            env_lines.extend(res)

        # -- YouTube Advanced Setup --
        setup_yt = await questionary.confirm(
            "Configure YouTube Professional Setup? (Recommended for stability)",
            default=True,
            style=QUESTIONARY_STYLE,
        ).ask_async()

        if setup_yt:
            console.print(
                "\n[info]YouTube now requires a PO Token and JS Runtime for stable extraction.[/]"
            )
            po_token = await questionary.text(
                "Enter YouTube PO Token (Optional, but recommended):",
                style=QUESTIONARY_STYLE,
            ).ask_async()
            if po_token:
                env_lines.append(f"AUDIORAG_YOUTUBE_PO_TOKEN={po_token}\n")

                # If PO token is provided, suggest visitor data for binding
                visitor_data = await questionary.text(
                    "Enter YouTube Visitor Data (Required if PO Token is bound to visitor session):",
                    style=QUESTIONARY_STYLE,
                ).ask_async()
                if visitor_data:
                    env_lines.append(f"AUDIORAG_YOUTUBE_VISITOR_DATA={visitor_data}\n")

                data_sync_id = await questionary.text(
                    "Enter YouTube Data Sync ID (Required if using PO Token with an account):",
                    style=QUESTIONARY_STYLE,
                ).ask_async()
                if data_sync_id:
                    env_lines.append(f"AUDIORAG_YOUTUBE_DATA_SYNC_ID={data_sync_id}\n")

            js_runtime = await questionary.select(
                "Select preferred JS Runtime (Required for YouTube):",
                choices=["deno", "node", "bun", "none"],
                default="deno",
                style=QUESTIONARY_STYLE,
            ).ask_async()
            if js_runtime and js_runtime != "none":
                env_lines.append(f"AUDIORAG_JS_RUNTIME={js_runtime}\n")

        new_config = {}
        for line in env_lines:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                new_config[key] = val

        env_path = Path(".env")
        final_lines = []

        if env_path.exists():
            with open(env_path) as f:
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


async def index_cmd(inputs: list[str], force: bool) -> None:
    config = AudioRAGConfig()

    async with AudioRAGPipeline(config) as pipeline:
        global _active_pipeline  # noqa: PLW0603
        _active_pipeline = pipeline

        try:
            with console.status("[info]Indexing sources...", spinner="dots"):
                result = await pipeline.index_many(inputs, force=force, raise_on_error=False)
        finally:
            _active_pipeline = None

    if not result.discovered_sources:
        console.print("[warning]No sources found to index.[/]")
        return

    source_count = len(result.discovered_sources)
    console.print(f"[highlight]Processed {source_count} discovered sources[/]")
    console.print(
        "[success]Indexed:[/] "
        f"{len(result.indexed_sources)}  "
        "[warning]Skipped:[/] "
        f"{len(result.skipped_sources)}  "
        "[error]Failed:[/] "
        f"{len(result.failures)}"
    )

    for failure in result.failures:
        console.print(
            f"[error]Failed:[/] {failure.source_url} "
            f"(stage={failure.stage}) - {failure.error_message}"
        )


async def query_cmd(query_text: str) -> None:
    """Perform a RAG query and display results in a top-tier layout."""
    config = AudioRAGConfig()

    async with AudioRAGPipeline(config) as pipeline:
        global _active_pipeline  # noqa: PLW0603
        _active_pipeline = pipeline

        try:
            with console.status("[info]Searching index...", spinner="dots"):
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
        finally:
            _active_pipeline = None


def _signal_handler(signum: int, _frame: Any) -> None:
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    console.print(f"\n[warning]Received {sig_name}, shutting down gracefully...[/]")
    sys.exit(0)


def main() -> None:
    """Entry point with clean help documentation."""
    parser = argparse.ArgumentParser(
        description="AudioRAG: Professional Audio RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  audiorag setup
  audiorag index "https://youtube.com/watch?v=..."
  audiorag index "./local_folder/"
  audiorag query "What are the main points?"

Note: Use "audiorag [command] --help" for more details on a specific command.
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Setup
    subparsers.add_parser("setup", help="Initialize provider configuration")

    # Index
    index_parser = subparsers.add_parser(
        "index",
        help="Process and index audio from URLs or paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: Always wrap URLs and paths with spaces in quotes to prevent shell errors.

Examples:
  audiorag index "https://youtube.com/watch?v=..."
  audiorag index "https://youtube.com/playlist?list=..."
  audiorag index "./audio_folder/" "local_file.mp3"
  audiorag index "My Music/Song.wav"
        """,
    )
    index_parser.add_argument("inputs", nargs="+", help="URLs or paths of the audio/video to index")
    index_parser.add_argument(
        "--force", action="store_true", help="Re-process even if already indexed"
    )

    # Query
    query_parser = subparsers.add_parser("query", help="Query the semantic index")
    query_parser.add_argument("text", help="The question to ask the audio index")

    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        if args.command == "setup":
            asyncio.run(setup_cmd())
        elif args.command == "index":
            asyncio.run(index_cmd(args.inputs, args.force))
        elif args.command == "query":
            asyncio.run(query_cmd(args.text))
        else:
            parser.print_help()
    except KeyboardInterrupt:
        console.print("\n[warning]Operation cancelled by user.[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()
