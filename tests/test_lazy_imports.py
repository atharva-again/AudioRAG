import builtins
import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("module_name", "dep_name"),
    [
        ("audiorag.embed.openai", "openai"),
        ("audiorag.embed.cohere", "cohere"),
        ("audiorag.embed.voyage", "voyageai"),
        ("audiorag.generate.openai", "openai"),
        ("audiorag.generate.anthropic", "anthropic"),
        ("audiorag.generate.gemini", "google.generativeai"),
        ("audiorag.transcribe.openai", "openai"),
        ("audiorag.transcribe.deepgram", "deepgram"),
        ("audiorag.transcribe.assemblyai", "assemblyai"),
        ("audiorag.transcribe.groq", "groq"),
        ("audiorag.store.chromadb", "chromadb"),
        ("audiorag.store.pinecone", "pinecone"),
        ("audiorag.store.weaviate", "weaviate"),
        ("audiorag.store.supabase", "vecs"),
        ("audiorag.source.url", "aiohttp"),
        ("audiorag.source.youtube", "yt_dlp"),
    ],
)
def test_lazy_import_module_imports_without_optional_dependency(
    module_name: str, dep_name: str, monkeypatch
) -> None:
    """Importing provider modules should not require optional dependencies."""
    if dep_name in sys.modules:
        del sys.modules[dep_name]

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == dep_name or name.startswith(f"{dep_name}."):
            raise ImportError(f"No module named {dep_name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    if module_name in sys.modules:
        del sys.modules[module_name]

    # Importing the module should not raise even if dependency is missing
    importlib.import_module(module_name)
