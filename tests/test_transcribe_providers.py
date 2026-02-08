"""Unit tests for transcription providers - instantiation tests."""

import pytest


class TestTranscriptionProvidersInstantiation:
    """Test transcription provider instantiation through lazy imports."""

    def test_openai_transcriber_instantiation(self) -> None:
        """Test OpenAI transcriber can be instantiated."""
        pytest.importorskip("openai")
        from audiorag.transcribe import OpenAITranscriber

        provider = OpenAITranscriber(api_key="test-key")
        assert provider.model == "whisper-1"

    def test_openai_transcriber_custom_model(self) -> None:
        """Test OpenAI transcriber with custom model."""
        pytest.importorskip("openai")
        from audiorag.transcribe import OpenAITranscriber

        provider = OpenAITranscriber(api_key="test-key", model="whisper-1")
        assert provider.model == "whisper-1"

    def test_assemblyai_transcriber_instantiation(self) -> None:
        """Test AssemblyAI transcriber can be instantiated."""
        pytest.importorskip("assemblyai")
        from audiorag.transcribe import AssemblyAITranscriber

        provider = AssemblyAITranscriber(api_key="test-key")
        assert provider._provider_name == "assemblyai_stt"

    def test_groq_transcriber_instantiation(self) -> None:
        """Test Groq transcriber can be instantiated."""
        pytest.importorskip("groq")
        from audiorag.transcribe import GroqTranscriber

        provider = GroqTranscriber(api_key="test-key")
        assert provider._provider_name == "groq_stt"

    def test_groq_transcriber_custom_model(self) -> None:
        """Test Groq transcriber with custom model."""
        pytest.importorskip("groq")
        from audiorag.transcribe import GroqTranscriber

        provider = GroqTranscriber(api_key="test-key", model="whisper-large-v3")
        assert provider.model == "whisper-large-v3"
