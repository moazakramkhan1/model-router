"""
providers/ollama_provider.py – async adapter for a locally-running Ollama instance.

API reference: POST {base_url}/api/generate (stream:false)
"""
import time
from typing import Any, Dict

import httpx

from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """Calls the Ollama /api/generate endpoint with stream disabled."""

    async def generate(self, user_input: str, max_tokens: int) -> Dict[str, Any]:
        base_url: str = self.config.get("base_url", "http://localhost:11434").rstrip("/")
        model: str = self.config.get("model", "llama3")
        timeout_s: float = float(self.config.get("timeout_s", 120))
        url = f"{base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": user_input,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)
            latency_ms = (time.monotonic() - t0) * 1000
            resp.raise_for_status()

        data: dict = resp.json()
        text: str = data.get("response", "")
        tokens: int = int(data.get("eval_count", 0) or 0)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "provider": "ollama",
        }
