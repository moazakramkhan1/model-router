"""
providers/__init__.py – provider registry and factory.
"""
from typing import Any, Dict

from .base import BaseProvider
from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider

# Map the string value of config["provider"] to its implementation class.
PROVIDER_REGISTRY: Dict[str, type] = {
    "ollama": OllamaProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}


def get_provider(slot_name: str, config: Dict[str, Any]) -> BaseProvider:
    """
    Instantiate and return the correct provider for a model slot.

    Parameters
    ----------
    slot_name : str   – logical slot key (e.g. "local_small")
    config    : dict  – slot config dict from models.yaml

    Raises
    ------
    ValueError if the provider type is not registered.
    """
    provider_type: str = config.get("provider", "")
    cls = PROVIDER_REGISTRY.get(provider_type)
    if cls is None:
        raise ValueError(
            f"Unknown provider type {provider_type!r} for slot {slot_name!r}. "
            f"Registered providers: {list(PROVIDER_REGISTRY)}"
        )
    return cls(slot_name=slot_name, config=config)
