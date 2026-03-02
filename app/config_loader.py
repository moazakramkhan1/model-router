"""
config_loader.py – loads config/models.yaml and config/policy.yaml,
expanding ${VAR:-default} / ${VAR} environment variable placeholders.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

# When running inside Docker the configs are volume-mounted at /app/config.
# Locally you can override with CONFIG_DIR env var.
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", "/app/config"))


# ---------------------------------------------------------------------------
# Env-var expansion helpers
# ---------------------------------------------------------------------------

_ENV_DEFAULT_RE = re.compile(r"\$\{([^}:]+):-([^}]*)\}")
_ENV_PLAIN_RE = re.compile(r"\$\{([^}]+)\}")


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
# Public loaders
# ---------------------------------------------------------------------------

def load_models() -> Dict[str, Any]:
    """Return the models registry dict keyed by slot name."""
    path = CONFIG_DIR / "models.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _expand(raw)["models"]


def load_policy() -> Dict[str, Any]:
    """Return the policy dict."""
    path = CONFIG_DIR / "policy.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _expand(raw)["policy"]
