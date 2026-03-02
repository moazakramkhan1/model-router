"""
providers/groq_provider.py – async adapter for any OpenAI-compatible chat endpoint.

Default target: POST https://api.groq.com/openai/v1/chat/completions
"""
import os
import time
from typing import Any, Dict

import httpx

from .base import BaseProvider


class GroqProvider(BaseProvider):
    """Async OpenAI-compatible adapter (default: api.groq.com)."""

    async def generate(self, user_input: str, max_tokens: int) -> Dict[str, Any]:
        base_url: str = self.config.get(
            "base_url", "https://api.groq.com/openai/v1"
        ).rstrip("/")
        model: str = self.config.get("model", "llama-3.1-8b-instant")
        api_key_env: str = self.config.get("api_key_env", "MID_API_KEY")
        api_key: str = os.environ.get(api_key_env, "")
        timeout_s: float = float(self.config.get("timeout_s", 60))

        if not api_key:
            raise ValueError(
                f"API key not set. Expected env var: {api_key_env!r}"
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
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
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
