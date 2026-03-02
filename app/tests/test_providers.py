"""Tests for provider adapters."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ollama_generate_success():
    from providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(
        "local_small",
        {"base_url": "http://localhost:11434", "model": "llama3", "timeout_s": 30},
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "The answer is 42.", "eval_count": 12}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("providers.ollama_provider.httpx.AsyncClient", return_value=mock_client):
        result = await provider.generate("What is 6×7?", 64)

    assert result["text"] == "The answer is 42."
    assert result["tokens"] == 12
    assert result["provider"] == "ollama"
    assert result["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_ollama_generate_http_error_raises():
    from providers.ollama_provider import OllamaProvider
    import httpx

    provider = OllamaProvider(
        "local_small",
        {"base_url": "http://localhost:11434", "model": "llama3", "timeout_s": 5},
    )

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with patch("providers.ollama_provider.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.ConnectError):
            await provider.generate("hello", 64)


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_groq_raises_without_api_key():
    from providers.groq_provider import GroqProvider

    os.environ.pop("MID_API_KEY", None)
    provider = GroqProvider(
        "fast_mid",
        {"base_url": "https://api.groq.com/openai/v1", "model": "llama3",
         "api_key_env": "MID_API_KEY", "timeout_s": 30},
    )
    with pytest.raises(ValueError, match="API key not set"):
        await provider.generate("hello", 64)


@pytest.mark.asyncio
async def test_groq_generate_success():
    from providers.groq_provider import GroqProvider

    os.environ["MID_API_KEY"] = "test-key"
    provider = GroqProvider(
        "fast_mid",
        {"base_url": "https://api.groq.com/openai/v1", "model": "llama3",
         "api_key_env": "MID_API_KEY", "timeout_s": 30},
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Bonjour!"}}],
        "usage": {"completion_tokens": 5},
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("providers.groq_provider.httpx.AsyncClient", return_value=mock_client):
        result = await provider.generate("Hello in French?", 64)

    assert result["text"] == "Bonjour!"
    assert result["tokens"] == 5
    assert result["provider"] == "groq"

    del os.environ["MID_API_KEY"]


# ---------------------------------------------------------------------------
# Circuit breaker unit tests
# ---------------------------------------------------------------------------

def test_circuit_breaker_opens_after_max_failures():
    from circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(fail_max=2, reset_timeout=60)
    assert cb.is_open("slot_a") is False

    cb.record_failure("slot_a")
    assert cb.is_open("slot_a") is False  # not yet

    cb.record_failure("slot_a")
    assert cb.is_open("slot_a") is True   # now open


def test_circuit_breaker_resets_on_success():
    from circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(fail_max=2, reset_timeout=60)
    cb.record_failure("slot_a")
    cb.record_failure("slot_a")
    assert cb.is_open("slot_a") is True

    cb.record_success("slot_a")
    assert cb.is_open("slot_a") is False


def test_circuit_breaker_half_open_after_timeout():
    import time
    from circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(fail_max=1, reset_timeout=0.01)  # 10 ms
    cb.record_failure("slot_b")
    assert cb.is_open("slot_b") is True

    time.sleep(0.02)
    # Should be half-open (let one probe through)
    assert cb.is_open("slot_b") is False
