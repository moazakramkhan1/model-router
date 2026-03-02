"""
providers/base.py – abstract async contract every provider adapter must implement.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseProvider(ABC):
    """
    A provider wraps one LLM backend (Ollama, Groq, Gemini, …).

    Parameters
    ----------
    slot_name : str
        The logical name of this model slot (from models.yaml).
    config : dict
        The slot's configuration dict (provider, model, base_url, api_key_env, …).
    """

    def __init__(self, slot_name: str, config: Dict[str, Any]) -> None:
        self.slot_name = slot_name
        self.config = config

    @abstractmethod
    async def generate(self, user_input: str, max_tokens: int) -> Dict[str, Any]:
        """
        Call the underlying model asynchronously and return a result dict.

        Returns
        -------
        dict with keys:
            text        : str   – the model's textual reply
            latency_ms  : float – wall-clock time for the call in milliseconds
            tokens      : int   – output token count (0 when unknown)
            provider    : str   – canonical provider name (e.g. "ollama")

        Raises
        ------
        Any exception on hard failure; the router will catch it.
        """
        ...
