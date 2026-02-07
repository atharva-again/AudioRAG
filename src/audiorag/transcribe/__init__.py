"""Transcription (STT) providers."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import with clear error messages for missing dependencies."""
    if name == "OpenAITranscriber":
        try:
            from audiorag.transcribe.openai import OpenAITranscriber

            return OpenAITranscriber
        except ImportError:
            raise ImportError(
                "OpenAITranscriber requires 'openai'. Install with: pip install audiorag[openai]"
            ) from None
    if name == "DeepgramTranscriber":
        try:
            from audiorag.transcribe.deepgram import DeepgramTranscriber

            return DeepgramTranscriber
        except ImportError:
            raise ImportError(
                "DeepgramTranscriber requires 'deepgram-sdk'. "
                "Install with: pip install audiorag[deepgram]"
            ) from None
    if name == "AssemblyAITranscriber":
        try:
            from audiorag.transcribe.assemblyai import AssemblyAITranscriber

            return AssemblyAITranscriber
        except ImportError:
            raise ImportError(
                "AssemblyAITranscriber requires 'assemblyai'. "
                "Install with: pip install audiorag[assemblyai]"
            ) from None
    if name == "GroqTranscriber":
        try:
            from audiorag.transcribe.groq import GroqTranscriber

            return GroqTranscriber
        except ImportError:
            raise ImportError(
                "GroqTranscriber requires 'groq'. Install with: pip install audiorag[groq]"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OpenAITranscriber",
    "DeepgramTranscriber",
    "AssemblyAITranscriber",
    "GroqTranscriber",
]
