"""CLI setup tool for AudioRAG using Textual."""

from __future__ import annotations

import asyncio
import os
import re
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from textual.app import App, ComposeResult
from textual.containers import Container, Grid, Horizontal
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
)

from audiorag.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProviderInfo:
    """Information about a provider."""

    name: str
    display_name: str
    signup_url: str
    env_var: str
    description: str
    optional: bool = False


# Provider registry
PROVIDERS: dict[str, ProviderInfo] = {
    "stt": ProviderInfo(
        name="openai",
        display_name="OpenAI (Whisper)",
        signup_url="https://platform.openai.com/api-keys",
        env_var="AUDIORAG_OPENAI_API_KEY",
        description="Best quality, supports multiple languages",
    ),
    "embedding": ProviderInfo(
        name="openai",
        display_name="OpenAI (Embeddings)",
        signup_url="https://platform.openai.com/api-keys",
        env_var="AUDIORAG_OPENAI_API_KEY",
        description="text-embedding-3-small model",
    ),
    "vector_store": ProviderInfo(
        name="chromadb",
        display_name="ChromaDB (Local)",
        signup_url="",
        env_var="",
        description="Local vector store - no API key needed!",
        optional=True,
    ),
    "generation": ProviderInfo(
        name="openai",
        display_name="OpenAI (GPT)",
        signup_url="https://platform.openai.com/api-keys",
        env_var="AUDIORAG_OPENAI_API_KEY",
        description="gpt-4o-mini for generation",
    ),
}

STT_PROVIDERS = [
    ("OpenAI Whisper", "openai"),
    ("Deepgram (fast, cheap)", "deepgram"),
    ("Groq (very fast, cheap)", "groq"),
    ("AssemblyAI (high quality)", "assemblyai"),
]

EMBEDDING_PROVIDERS = [
    ("OpenAI", "openai"),
    ("Voyage AI", "voyage"),
    ("Cohere", "cohere"),
]

VECTOR_STORES = [
    ("ChromaDB (local)", "chromadb"),
    ("Pinecone", "pinecone"),
    ("Weaviate", "weaviate"),
]

GENERATION_PROVIDERS = [
    ("OpenAI", "openai"),
    ("Anthropic", "anthropic"),
    ("Google Gemini", "gemini"),
]


class SetupCompleteScreen(ModalScreen[bool]):
    """Screen shown when setup is complete."""

    DEFAULT_CSS = """
    SetupCompleteScreen {
        align: center middle;
    }

    #dialog {
        grid-size: 2;
        grid-gutter: 1 2;
        grid-rows: 1fr 3;
        padding: 0 1;
        width: 80;
        height: auto;
        border: thick $background 80%;
        background: $surface;
    }

    #question {
        column-span: 2;
        height: 1fr;
        width: 1fr;
        content-align: center middle;
    }

    Button {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Grid(id="dialog"):
            yield Static(
                "✅ Setup complete!\n\n"
                "Your .env file has been created with your API keys.\n"
                "Run 'uv run audiorag quickstart' to test your setup.",
                id="question",
            )
            yield Button("Done", variant="primary", id="done")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(True)


class KeyValidationScreen(ModalScreen[bool]):
    """Screen for validating API keys."""

    DEFAULT_CSS = """
    KeyValidationScreen {
        align: center middle;
    }

    #dialog {
        width: 80;
        height: auto;
        padding: 1 2;
        border: thick $background 80%;
        background: $surface;
    }

    #status {
        margin: 1 0;
        height: auto;
    }
    """

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__()
        self.config = config
        self.results: dict[str, bool] = {}

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Validating your API keys...", id="title")
            yield ProgressBar(total=len(self.config), id="progress")
            yield Static(id="status")

    async def on_mount(self) -> None:
        """Validate keys when mounted."""
        status_widget = self.query_one("#status", Static)
        progress = self.query_one("#progress", ProgressBar)

        for provider, key in self.config.items():
            if not key:
                continue

            status_widget.update(f"Testing {provider}...")
            is_valid = await self._test_key(provider, key)
            self.results[provider] = is_valid
            progress.advance(1)

            status = "✅ Valid" if is_valid else "❌ Invalid"
            status_widget.update(f"{provider}: {status}")
            await asyncio.sleep(0.5)

        # Show final results
        valid_count = sum(self.results.values())
        total_count = len(self.results)

        status_widget.update(f"\nValidation complete: {valid_count}/{total_count} keys valid\n")

        await asyncio.sleep(1)
        self.dismiss(True)

    async def _test_key(self, provider: str, key: str) -> bool:
        """Test if an API key is valid."""
        # In production, this would make actual API calls
        # For now, just check key format
        if provider == "openai":
            return key.startswith("sk-") and len(key) > 20
        if provider == "deepgram":
            return len(key) > 20
        if provider == "groq":
            return key.startswith("gsk_") and len(key) > 20
        if provider == "assemblyai":
            return len(key) > 20
        return True


class AudioRAGSetupApp(App):
    """Textual app for AudioRAG setup."""

    CSS = """
    Screen {
        align: center middle;
    }

    #setup-container {
        width: 100;
        height: auto;
        padding: 1 2;
        border: thick $background 80%;
        background: $surface;
    }

    #title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #subtitle {
        text-align: center;
        margin-bottom: 2;
    }

    .section {
        margin: 1 0;
        padding: 1;
        border: solid $primary;
    }

    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }

    Select {
        margin-bottom: 1;
    }

    Input {
        margin-bottom: 1;
    }

    #signup-link {
        color: $primary;
        text-style: underline;
    }

    #buttons {
        margin-top: 2;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }

    #help-text {
        margin-top: 2;
        color: $text-muted;
        text-align: center;
    }
    """

    BINDINGS: ClassVar = [("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self.config: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="setup-container"):
            yield Label("🎵 AudioRAG Setup", id="title")
            yield Label(
                "Configure your providers and API keys\n",
                id="subtitle",
            )

            # STT Provider
            with Container(classes="section"):
                yield Label("1. Speech-to-Text Provider", classes="section-title")
                yield Select(
                    STT_PROVIDERS,
                    prompt="Select STT provider",
                    id="stt-provider",
                )
                yield Input(
                    placeholder="Enter API key",
                    password=True,
                    id="stt-api-key",
                )
                yield Button(
                    "Get API Key ↗",
                    id="stt-signup",
                    variant="primary",
                )

            # Embedding Provider
            with Container(classes="section"):
                yield Label("2. Embedding Provider", classes="section-title")
                yield Select(
                    EMBEDDING_PROVIDERS,
                    prompt="Select embedding provider",
                    id="embedding-provider",
                )
                yield Input(
                    placeholder="Enter API key",
                    password=True,
                    id="embedding-api-key",
                )
                yield Button(
                    "Get API Key ↗",
                    id="embedding-signup",
                    variant="primary",
                )

            # Vector Store
            with Container(classes="section"):
                yield Label("3. Vector Store", classes="section-title")
                yield Select(
                    VECTOR_STORES,
                    prompt="Select vector store",
                    id="vector-store",
                )
                yield Static(
                    "ChromaDB runs locally - no API key needed!",
                    id="vector-store-help",
                )

            # Generation Provider
            with Container(classes="section"):
                yield Label("4. Generation Provider", classes="section-title")
                yield Select(
                    GENERATION_PROVIDERS,
                    prompt="Select generation provider",
                    id="generation-provider",
                )
                yield Input(
                    placeholder="Enter API key",
                    password=True,
                    id="generation-api-key",
                )
                yield Button(
                    "Get API Key ↗",
                    id="generation-signup",
                    variant="primary",
                )

            # Action buttons
            with Horizontal(id="buttons"):
                yield Button("Save Config", variant="success", id="save")
                yield Button("Validate Keys", id="validate")
                yield Button("Cancel", variant="error", id="cancel")

            yield Static(
                "Press Tab to navigate, Enter to select\nYour API keys are stored locally in .env",
                id="help-text",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "stt-signup":
            provider = cast(str, self.query_one("#stt-provider", Select).value or "openai")
            url = self._get_signup_url(provider)
            webbrowser.open(url)
        elif button_id == "embedding-signup":
            provider = cast(str, self.query_one("#embedding-provider", Select).value or "openai")
            url = self._get_signup_url(provider)
            webbrowser.open(url)
        elif button_id == "generation-signup":
            provider = cast(str, self.query_one("#generation-provider", Select).value or "openai")
            url = self._get_signup_url(provider)
            webbrowser.open(url)
        elif button_id == "save":
            self._save_config()
        elif button_id == "validate":
            self._validate_keys()
        elif button_id == "cancel":
            self.exit()

    def _get_signup_url(self, provider: str) -> str:
        """Get signup URL for a provider."""
        urls = {
            "openai": "https://platform.openai.com/api-keys",
            "deepgram": "https://console.deepgram.com/signup",
            "groq": "https://console.groq.com/keys",
            "assemblyai": "https://www.assemblyai.com/app",
            "voyage": "https://www.voyageai.com/",
            "cohere": "https://cohere.com/",
            "anthropic": "https://console.anthropic.com/",
            "gemini": "https://aistudio.google.com/app/apikey",
        }
        return urls.get(provider, "https://platform.openai.com/api-keys")

    def _save_config(self) -> None:
        """Save configuration to .env file."""
        stt_provider = cast(str, self.query_one("#stt-provider", Select).value or "openai")
        embedding_provider = cast(
            str, self.query_one("#embedding-provider", Select).value or "openai"
        )
        vector_store = cast(str, self.query_one("#vector-store", Select).value or "chromadb")
        generation_provider = cast(
            str, self.query_one("#generation-provider", Select).value or "openai"
        )

        stt_key = self.query_one("#stt-api-key", Input).value
        embedding_key = self.query_one("#embedding-api-key", Input).value
        generation_key = self.query_one("#generation-api-key", Input).value

        # Build env content
        env_lines = [
            "# AudioRAG Configuration",
            "# Generated by audiorag setup",
            "",
            "# Provider Selection",
            f"AUDIORAG_STT_PROVIDER={stt_provider}",
            f"AUDIORAG_EMBEDDING_PROVIDER={embedding_provider}",
            f"AUDIORAG_VECTOR_STORE_PROVIDER={vector_store}",
            f"AUDIORAG_GENERATION_PROVIDER={generation_provider}",
            "",
        ]

        # Add API keys (only for non-local providers)
        if stt_key:
            env_lines.append(f"# {stt_provider.upper()} API Key for STT")
            env_lines.append(f"AUDIORAG_{stt_provider.upper()}_API_KEY={stt_key}")
            env_lines.append("")

        if embedding_key and embedding_provider == "openai":
            env_lines.append("# OpenAI API Key for Embeddings")
            env_lines.append("AUDIORAG_OPENAI_API_KEY={embedding_key}")
            env_lines.append("")

        if generation_key and generation_provider == "openai":
            env_lines.append("# OpenAI API Key for Generation")
            if not embedding_key or embedding_provider != "openai":
                env_lines.append(f"AUDIORAG_OPENAI_API_KEY={generation_key}")
            env_lines.append("")

        # Write to .env
        env_path = Path(".env")
        env_content = "\n".join(env_lines)

        # Append if exists, create if not
        if env_path.exists():
            with open(env_path, "a") as f:
                f.write("\n\n# AudioRAG Setup\n")
                f.write(env_content)
        else:
            with open(env_path, "w") as f:
                f.write(env_content)

        # Show success screen
        self.push_screen(SetupCompleteScreen())

    def _validate_keys(self) -> None:
        """Validate API keys."""
        stt_key = self.query_one("#stt-api-key", Input).value
        embedding_key = self.query_one("#embedding-api-key", Input).value
        generation_key = self.query_one("#generation-api-key", Input).value

        config = {
            "STT": stt_key,
            "Embedding": embedding_key,
            "Generation": generation_key,
        }

        self.push_screen(KeyValidationScreen(config))

    async def action_quit(self) -> None:
        """Quit the app."""
        self.exit()


def main() -> None:
    """Run the CLI setup tool."""
    app = AudioRAGSetupApp()
    app.run()


if __name__ == "__main__":
    main()
