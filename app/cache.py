"""
cache.py – Redis-backed prompt result cache.

Cache key is a SHA-256 hash of (user_input + require_json flag) so
semantically identical requests return cached results without hitting
any provider.

Redis is optional: if REDIS_URL is not set or Redis is unreachable
every function silently returns None / no-ops.
"""
import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

CACHE_TTL: int = int(os.environ.get("CACHE_TTL_SECONDS", "300"))  # 5 minutes default

# Lazy singleton – avoids import-time connection attempt
_redis_client = None


def _get_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        import redis.asyncio as aioredis  # type: ignore
        _redis_client = aioredis.from_url(url, decode_responses=True)
        return _redis_client
    except Exception as exc:
        log.warning("Redis client init failed: %s", exc)
        return None


def _cache_key(user_input: str, require_json: bool) -> str:
    raw = f"{user_input}|{require_json}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return f"router:cache:{digest}"


async def get_cached(user_input: str, require_json: bool) -> Optional[Dict[str, Any]]:
    """Return cached result or None if not cached / Redis unavailable."""
    client = _get_client()
    if client is None:
        return None
    try:
        raw = await client.get(_cache_key(user_input, require_json))
        if raw:
            return json.loads(raw)
    except Exception as exc:
        log.debug("Cache get failed: %s", exc)
    return None


async def set_cached(
    user_input: str, require_json: bool, result: Dict[str, Any]
) -> None:
    """Persist a successful result with TTL. Silently ignores errors."""
    client = _get_client()
    if client is None:
        return
    try:
        await client.setex(
            _cache_key(user_input, require_json),
            CACHE_TTL,
            json.dumps(result),
        )
    except Exception as exc:
        log.debug("Cache set failed: %s", exc)
