"""
providers/groq_provider.py – adapter for Groq's OpenAI-compatible chat endpoint.

Endpoint: POST https://api.groq.com/openai/v1/chat/completions
"""
import os
import time
from typing import Any, Dict

import requests

from .base import BaseProvider


class GroqProvider(BaseProvider):
    """OpenAI-compatible adapter targeting api.groq.com."""

    def generate(self, user_input: str, max_tokens: int) -> Dict[str, Any]:
        base_url: str = self.config.get(
            "base_url", "https://api.groq.com/openai/v1"
        ).rstrip("/")
        model: str = self.config.get("model", "llama-3.1-8b-instant")
        api_key_env: str = self.config.get("api_key_env", "GROQ_API_KEY")
        api_key: str = os.environ.get(api_key_env, "")

        if not api_key:
            raise ValueError(
                f"Groq API key not set. Expected env var: {api_key_env!r}"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_input}],
            "max_tokens": max_tokens,
        }

        t0 = time.monotonic()
        resp = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        resp.raise_for_status()
        data: dict = resp.json()

        text: str = data["choices"][0]["message"]["content"]
        usage: dict = data.get("usage", {})
        tokens: int = int(usage.get("completion_tokens", 0) or 0)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "provider": "groq",
        }
