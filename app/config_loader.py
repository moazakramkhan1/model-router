"""
config_loader.py – loads config/models.yaml and config/policy.yaml,
expanding ${VAR:-default} / ${VAR} environment variable placeholders.

Features
--------
- In-memory cache so configs are only read from disk once per process
- hot-reload via watchfiles (watch_configs() background coroutine)
- Thread-safe cache invalidation via dict.clear() + dict.update()
"""
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", "/app/config"))
log = logging.getLogger(__name__)

_ENV_DEFAULT_RE = re.compile(r"\$\{([^}:]+):-([^}]*)\}")
_ENV_PLAIN_RE = re.compile(r"\$\{([^}]+)\}")

# In-memory caches
_models_cache: Dict[str, Any] = {}
_policy_cache: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Env-var expansion helpers
# ---------------------------------------------------------------------------

def _expand_str(value: str) -> str:
    """Expand ${VAR:-default} then ${VAR} patterns in a string."""
    def _replace_with_default(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        return os.environ.get(var, default)

    def _replace_plain(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")

    value = _ENV_DEFAULT_RE.sub(_replace_with_default, value)
    value = _ENV_PLAIN_RE.sub(_replace_plain, value)
    return value


def _expand(node: Any) -> Any:
    """Recursively expand env-vars in all string leaves of a parsed YAML tree."""
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(item) for item in node]
    if isinstance(node, str):
        return _expand_str(node)
    return node


# ---------------------------------------------------------------------------
# Public loaders (cached)
# ---------------------------------------------------------------------------

def load_models() -> Dict[str, Any]:
    """Return the models registry dict keyed by slot name."""
    if not _models_cache:
        path = CONFIG_DIR / "models.yaml"
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        _models_cache.update(_expand(raw)["models"])
    return _models_cache


def load_policy() -> Dict[str, Any]:
    """Return the policy dict."""
    if not _policy_cache:
        path = CONFIG_DIR / "policy.yaml"
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        _policy_cache.update(_expand(raw)["policy"])
    return _policy_cache


def reload_configs() -> None:
    """Invalidate caches and force an immediate reload from disk."""
    _models_cache.clear()
    _policy_cache.clear()
    load_models()
    load_policy()
    log.info("Configs reloaded from %s", CONFIG_DIR)


# ---------------------------------------------------------------------------
# Hot-reload background coroutine
# ---------------------------------------------------------------------------

async def watch_configs() -> None:
    """
    Watch CONFIG_DIR for file changes and reload configs automatically.
    Run as an asyncio background task from main.py lifespan.
    Requires the `watchfiles` package; silently disables itself if absent.
    """
    try:
        from watchfiles import awatch  # type: ignore
        log.info("Config hot-reload watcher started (watching %s).", CONFIG_DIR)
        async for _ in awatch(str(CONFIG_DIR)):
            log.info("Config change detected – reloading.")
            reload_configs()
    except ImportError:
        log.info("watchfiles not installed; config hot-reload disabled.")
    except asyncio.CancelledError:
        log.info("Config watcher stopped.")
    except Exception as exc:  # noqa: BLE001
        log.error("Config watcher error: %s", exc)


def load_policy() -> Dict[str, Any]:
    """Return the policy dict."""
    path = CONFIG_DIR / "policy.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _expand(raw)["policy"]
