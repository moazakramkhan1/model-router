"""
providers/gemini_provider.py – async adapter for Google Gemini via google-genai SDK.

The google-genai SDK is synchronous, so calls are dispatched to a thread-pool
executor via asyncio.to_thread() to avoid blocking the event loop.
"""
import asyncio
import os
import time
from typing import Any, Dict

from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Async wrapper around the google-genai synchronous SDK."""

    async def generate(self, user_input: str, max_tokens: int) -> Dict[str, Any]:
        try:
            from google import genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-genai package is not installed. "
                "Add google-genai to requirements.txt."
            ) from exc

        api_key_env: str = self.config.get("api_key_env", "PREMIUM_API_KEY")
        api_key: str = os.environ.get(api_key_env, "")

        if not api_key:
            raise ValueError(
                f"API key not set. Expected env var: {api_key_env!r}"
            )

        model: str = self.config.get("model", "gemini-1.5-flash")

        def _sync_call() -> tuple:
            client = genai.Client(api_key=api_key)
            gen_config = genai_types.GenerateContentConfig(
                max_output_tokens=max_tokens,
            )
            t0 = time.monotonic()
            response = client.models.generate_content(
                model=model,
                contents=user_input,
                config=gen_config,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            text = response.text or ""
            tokens = 0
            try:
                if response.usage_metadata:
                    tokens = int(
                        getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                    )
            except Exception:
                pass
            return text, latency_ms, tokens

        text, latency_ms, tokens = await asyncio.to_thread(_sync_call)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "provider": "gemini",
        }
