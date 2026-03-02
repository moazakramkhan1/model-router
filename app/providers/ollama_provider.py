"""
providers/ollama_provider.py – adapter for a locally-running Ollama instance.

API reference: POST {base_url}/api/generate (stream:false)
"""
import time
from typing import Any, Dict

import requests

from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """Calls the Ollama /api/generate endpoint with stream disabled."""

    def generate(self, user_input: str, max_tokens: int) -> Dict[str, Any]:
        base_url: str = self.config.get("base_url", "http://localhost:11434").rstrip("/")
        model: str = self.config.get("model", "llama3")
        url = f"{base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": user_input,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }

        t0 = time.monotonic()
        resp = requests.post(url, json=payload, timeout=120)
        latency_ms = (time.monotonic() - t0) * 1000

        resp.raise_for_status()
        data: dict = resp.json()

        text: str = data.get("response", "")
        # eval_count = number of tokens in the generated response
        tokens: int = int(data.get("eval_count", 0) or 0)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "provider": "ollama",
        }
