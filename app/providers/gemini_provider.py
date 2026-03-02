"""
providers/gemini_provider.py – adapter for Google Gemini via the google-genai SDK.
"""
import os
import time
from typing import Any, Dict

from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Uses google-genai SDK to call the Gemini API."""

    def generate(self, user_input: str, max_tokens: int) -> Dict[str, Any]:
        try:
            from google import genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-genai package is not installed. "
                "Add google-genai to requirements.txt."
            ) from exc

        api_key_env: str = self.config.get("api_key_env", "GEMINI_API_KEY")
        api_key: str = os.environ.get(api_key_env, "")

        if not api_key:
            raise ValueError(
                f"Gemini API key not set. Expected env var: {api_key_env!r}"
            )

        model: str = self.config.get("model", "gemini-1.5-flash")
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

        text: str = response.text or ""

        tokens: int = 0
        try:
            usage = response.usage_metadata
            if usage:
                tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        except Exception:
            tokens = 0

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "provider": "gemini",
        }
