"""
router.py – core routing and cost-optimisation logic.

Flow for a single task
----------------------
1. Check Redis cache; return immediately on hit.
2. Load models + policy configs.
3. For each slot in the path (up to max_attempts, within budget):
   a. Skip if circuit breaker is open.
   b. Instantiate provider.
   c. Call provider.generate() with transient-error retry.
   d. Run async judge (rules + tiny LLM) on the result.
   e. Record attempt + Prometheus metrics.
   f. If ok → cache result, break, return success.
4. Return final result regardless of outcome.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import structlog

from cache import get_cached, set_cached
from circuit_breaker import circuit_breaker
from config_loader import load_models, load_policy
from eval import judge
from metrics import (
    router_attempts_total,
    router_latency_ms,
    router_tasks_total,
    router_virtual_cost_units_total,
)
from providers import get_provider

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Difficulty heuristic – token-count based
# ---------------------------------------------------------------------------

_HARD_TOKEN_THRESHOLD = 75  # estimated input tokens


def _estimate_input_tokens(text: str) -> float:
    """Rough token estimate: word count × 1.3 (subword tokenisation proxy)."""
    return len(text.split()) * 1.3


def _estimate_difficulty(user_input: str) -> str:
    """Return 'hard' when the prompt exceeds ~58 words, else 'easy'."""
    return "hard" if _estimate_input_tokens(user_input) >= _HARD_TOKEN_THRESHOLD else "easy"


def _max_tokens_for_difficulty(difficulty: str) -> int:
    return 1024 if difficulty == "hard" else 512


# ---------------------------------------------------------------------------
# Virtual cost computation
# ---------------------------------------------------------------------------

def _compute_virtual_cost(tokens: int, cost_per_1k: float) -> int:
    if cost_per_1k <= 0:
        return 0
    if tokens == 0:
        return 1
    return int(tokens / 1000 * cost_per_1k) + 1


# ---------------------------------------------------------------------------
# Transient-error retry helper
# ---------------------------------------------------------------------------

# HTTP status codes that warrant a retry (rate-limit + server errors)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


async def _call_with_retry(
    provider, user_input: str, max_tokens: int
) -> Dict[str, Any]:
    """
    Call provider.generate() with exponential back-off on transient errors.
    Non-retryable errors (e.g. auth failures, bad requests) are raised immediately.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt_num in range(1, 4):  # max 3 internal retries
        try:
            return await provider.generate(user_input, max_tokens)
        except Exception as exc:
            last_exc = exc
            status_code = getattr(
                getattr(exc, "response", None), "status_code", None
            )
            if status_code in _RETRYABLE_STATUS:
                wait = 2 ** (attempt_num - 1)  # 1 s, 2 s, 4 s
                log.warning(
                    "provider_retry",
                    attempt=attempt_num,
                    wait_s=wait,
                    error=str(exc)[:80],
                )
                await asyncio.sleep(wait)
                continue
            raise  # non-retryable – propagate immediately
    raise last_exc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_task(
    task_id: str,
    user_input: str,
    task_type: Optional[str],
    risk: str,
    budget_virtual_units: int,
    max_latency_ms: float,
    require_json: bool,
) -> Dict[str, Any]:
    """
    Route a task through the configured model path and return the full result.

    Returns
    -------
    {
        success                 : bool
        final_model             : str | None
        final_text              : str | None
        total_virtual_cost_units: int
        attempts                : List[dict]
        cached                  : bool
    }
    """
    # ─ Cache check ────────────────────────────────────────────────────────────
    cached = await get_cached(user_input, require_json)
    if cached is not None:
        log.info("cache_hit", task_id=task_id)
        cached["cached"] = True
        return cached

    models_cfg: Dict[str, Any] = load_models()
    policy_cfg: Dict[str, Any] = load_policy()

    path_key = "low_risk_path" if risk == "low" else "high_risk_path"
    model_path: List[str] = policy_cfg.get(path_key, [])
    max_attempts: int = int(policy_cfg.get("max_attempts", 3))
    min_score: float = float(policy_cfg.get("min_score", 0.5))

    difficulty = _estimate_difficulty(user_input)
    max_tokens = _max_tokens_for_difficulty(difficulty)

    log.info(
        "task_start",
        task_id=task_id,
        risk=risk,
        difficulty=difficulty,
        max_tokens=max_tokens,
        path=model_path,
    )

    attempts: List[Dict[str, Any]] = []
    total_cost: int = 0
    success: bool = False
    final_model: Optional[str] = None
    final_text: Optional[str] = None

    for slot_name in model_path[:max_attempts]:
        # ─ Budget gate ─────────────────────────────────────────────────────────
        if total_cost >= budget_virtual_units:
            log.info(
                "budget_exhausted",
                task_id=task_id,
                spent=total_cost,
                budget=budget_virtual_units,
            )
            break

        # ─ Circuit breaker ─────────────────────────────────────────────────────
        if circuit_breaker.is_open(slot_name):
            log.warning("circuit_open", task_id=task_id, slot=slot_name)
            attempts.append({
                "slot": slot_name,
                "provider": "unknown",
                "ok": False,
                "score": 0.0,
                "reasons": ["circuit_breaker_open"],
                "latency_ms": 0.0,
                "tokens": 0,
                "virtual_cost_units": 0,
                "text": "",
            })
            continue

        slot_cfg = models_cfg.get(slot_name)
        if slot_cfg is None:
            log.warning("slot_not_found", task_id=task_id, slot=slot_name)
            continue

        cost_per_1k: float = float(slot_cfg.get("virtual_cost_per_1k", 0))

        attempt: Dict[str, Any] = {
            "slot": slot_name,
            "provider": slot_cfg.get("provider", "unknown"),
            "ok": False,
            "score": 0.0,
            "reasons": [],
            "latency_ms": 0.0,
            "tokens": 0,
            "virtual_cost_units": 0,
            "text": "",
        }

        t0 = time.monotonic()
        try:
            provider = get_provider(slot_name, slot_cfg)
            result = await _call_with_retry(provider, user_input, max_tokens)
            latency_ms = (time.monotonic() - t0) * 1000

            attempt["latency_ms"] = round(latency_ms, 2)
            attempt["tokens"] = int(result.get("tokens", 0) or 0)
            attempt["text"] = result.get("text", "")
            attempt["provider"] = result.get(
                "provider", slot_cfg.get("provider", "unknown")
            )

            # ─ Latency gate ──────────────────────────────────────────────────
            if latency_ms > max_latency_ms:
                attempt["ok"] = False
                attempt["reasons"] = [
                    f"latency_exceeded:{latency_ms:.0f}ms>limit:{max_latency_ms:.0f}ms"
                ]
                attempt["score"] = 0.0
            else:
                # ─ Async judge (rules + tiny LLM) ────────────────────────────
                eval_result = await judge(
                    text=attempt["text"],
                    require_json=require_json,
                    user_input=user_input,
                )
                # Apply min_score threshold on top of rule-based ok flag
                attempt["ok"] = eval_result["ok"] and eval_result["score"] >= min_score
                attempt["score"] = eval_result["score"]
                attempt["reasons"] = eval_result["reasons"]

            circuit_breaker.record_success(slot_name)

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - t0) * 1000
            attempt["latency_ms"] = round(latency_ms, 2)
            attempt["ok"] = False
            attempt["reasons"] = [f"{type(exc).__name__}:{str(exc)[:120]}"]
            attempt["score"] = 0.0
            circuit_breaker.record_failure(slot_name)
            log.warning(
                "provider_error",
                task_id=task_id,
                slot=slot_name,
                error_type=type(exc).__name__,
                error=str(exc)[:120],
            )

        # ─ Cost accounting ──────────────────────────────────────────────────────
        vc = _compute_virtual_cost(attempt["tokens"], cost_per_1k)
        attempt["virtual_cost_units"] = vc
        total_cost += vc

        # ─ Prometheus metrics ────────────────────────────────────────────────
        ok_label = "true" if attempt["ok"] else "false"
        router_attempts_total.labels(model=slot_name, ok=ok_label).inc()
        router_virtual_cost_units_total.labels(model=slot_name).inc(vc)
        router_latency_ms.labels(model=slot_name).observe(attempt["latency_ms"])

        attempts.append(attempt)

        if attempt["ok"]:
            success = True
            final_model = slot_name
            final_text = attempt["text"]
            log.info(
                "task_success",
                task_id=task_id,
                slot=slot_name,
                score=attempt["score"],
                cost=vc,
                total_cost=total_cost,
            )
            break
        else:
            log.info(
                "attempt_failed",
                task_id=task_id,
                slot=slot_name,
                score=attempt["score"],
                reasons=attempt["reasons"],
            )

    # ─ Task-level Prometheus metric ───────────────────────────────────────────
    router_tasks_total.labels(success="true" if success else "false").inc()

    result = {
        "success": success,
        "final_model": final_model,
        "final_text": final_text,
        "total_virtual_cost_units": total_cost,
        "attempts": attempts,
        "cached": False,
    }

    # Cache successful results for future identical prompts
    if success:
        await set_cached(user_input, require_json, result)

    log.info(
        "task_done",
        task_id=task_id,
        success=success,
        final_model=final_model,
        total_cost=total_cost,
        num_attempts=len(attempts),
    )
    return result

# ---------------------------------------------------------------------------
# Difficulty heuristic
# ---------------------------------------------------------------------------

# Prompts whose estimated token count exceeds this threshold are treated as
# hard regardless of keyword presence.  The multiplier 1.3 approximates
# subword tokenisation (words are split into ~1.3 tokens on average).
_HARD_TOKEN_THRESHOLD = 75  # estimated input tokens


def _estimate_input_tokens(text: str) -> float:
    """Rough token estimate: word count × 1.3 (subword tokenisation proxy)."""
    return len(text.split()) * 1.3


def _estimate_difficulty(user_input: str) -> str:
    """
    Return 'hard' if the prompt is long (estimated ≥ 75 tokens) OR if its
    word count alone signals a complex task, else 'easy'.
    """
    if _estimate_input_tokens(user_input) >= _HARD_TOKEN_THRESHOLD:
        return "hard"
    return "easy"


def _max_tokens_for_difficulty(difficulty: str) -> int:
    return 1024 if difficulty == "hard" else 512


# ---------------------------------------------------------------------------
# Virtual cost computation
# ---------------------------------------------------------------------------

def _compute_virtual_cost(tokens: int, cost_per_1k: float) -> int:
    """
    Rules from the spec:
     - free providers (cost_per_1k == 0)  → always 0
     - tokens unknown (== 0) and paid     → charge minimum 1 unit
     - otherwise: floor(tokens/1000 * cost_per_1k) + 1
    """
    if cost_per_1k <= 0:
        return 0
    if tokens == 0:
        return 1
    return int(tokens / 1000 * cost_per_1k) + 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_task(
    task_id: str,
    user_input: str,
    task_type: Optional[str],
    risk: str,
    budget_virtual_units: int,
    max_latency_ms: float,
    require_json: bool,
) -> Dict[str, Any]:
    """
    Route a task through the configured model path and return the full result.

    Returns
    -------
    {
        success                : bool
        final_model            : str | None
        final_text             : str | None
        total_virtual_cost_units: int
        attempts               : List[dict]
    }
    """
    models_cfg: Dict[str, Any] = load_models()
    policy_cfg: Dict[str, Any] = load_policy()

    path_key = "low_risk_path" if risk == "low" else "high_risk_path"
    model_path: List[str] = policy_cfg.get(path_key, [])
    max_attempts: int = int(policy_cfg.get("max_attempts", 3))

    difficulty = _estimate_difficulty(user_input)
    max_tokens = _max_tokens_for_difficulty(difficulty)

    logger.info(
        "task_id=%s risk=%s difficulty=%s max_tokens=%d path=%s",
        task_id, risk, difficulty, max_tokens, model_path,
    )

    attempts: List[Dict[str, Any]] = []
    total_cost: int = 0
    success: bool = False
    final_model: Optional[str] = None
    final_text: Optional[str] = None

    for slot_name in model_path[:max_attempts]:
        # Budget gate
        if total_cost >= budget_virtual_units:
            logger.info(
                "task_id=%s budget exhausted (%d>=%d), stopping.",
                task_id, total_cost, budget_virtual_units,
            )
            break

        slot_cfg = models_cfg.get(slot_name)
        if slot_cfg is None:
            logger.warning(
                "task_id=%s slot %r not found in models.yaml – skipping.",
                task_id, slot_name,
            )
            continue

        cost_per_1k: float = float(slot_cfg.get("virtual_cost_per_1k", 0))

        attempt: Dict[str, Any] = {
            "slot": slot_name,
            "provider": slot_cfg.get("provider", "unknown"),
            "ok": False,
            "score": 0.0,
            "reasons": [],
            "latency_ms": 0.0,
            "tokens": 0,
            "virtual_cost_units": 0,
            "text": "",
        }

        t0 = time.monotonic()
        try:
            provider = get_provider(slot_name, slot_cfg)
            result = provider.generate(user_input, max_tokens)
            latency_ms = (time.monotonic() - t0) * 1000

            attempt["latency_ms"] = round(latency_ms, 2)
            attempt["tokens"] = int(result.get("tokens", 0) or 0)
            attempt["text"] = result.get("text", "")
            attempt["provider"] = result.get("provider", slot_cfg.get("provider", "unknown"))

            # Latency gate (evaluate regardless – still record cost)
            if latency_ms > max_latency_ms:
                attempt["ok"] = False
                attempt["reasons"] = [
                    f"latency_exceeded:{latency_ms:.0f}ms_limit:{max_latency_ms:.0f}ms"
                ]
                attempt["score"] = 0.0
                logger.info(
                    "task_id=%s slot=%s latency %.0fms > limit %.0fms",
                    task_id, slot_name, latency_ms, max_latency_ms,
                )
            else:
                eval_result = judge(attempt["text"], require_json=require_json)
                attempt["ok"] = eval_result["ok"]
                attempt["score"] = eval_result["score"]
                attempt["reasons"] = eval_result["reasons"]

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - t0) * 1000
            attempt["latency_ms"] = round(latency_ms, 2)
            attempt["ok"] = False
            attempt["reasons"] = [f"{type(exc).__name__}:{str(exc)[:120]}"]
            attempt["score"] = 0.0
            logger.warning(
                "task_id=%s slot=%s provider error: %s: %s",
                task_id, slot_name, type(exc).__name__, exc,
            )

        # Compute and accumulate virtual cost
        vc = _compute_virtual_cost(attempt["tokens"], cost_per_1k)
        attempt["virtual_cost_units"] = vc
        total_cost += vc

        # ── Prometheus metrics ──────────────────────────────────────────────
        ok_label = "true" if attempt["ok"] else "false"
        router_attempts_total.labels(model=slot_name, ok=ok_label).inc()
        router_virtual_cost_units_total.labels(model=slot_name).inc(vc)
        router_latency_ms.labels(model=slot_name).observe(attempt["latency_ms"])

        attempts.append(attempt)

        if attempt["ok"]:
            success = True
            final_model = slot_name
            final_text = attempt["text"]
            logger.info(
                "task_id=%s succeeded on slot=%s cost=%d total_cost=%d",
                task_id, slot_name, vc, total_cost,
            )
            break
        else:
            logger.info(
                "task_id=%s slot=%s failed (reasons=%s), trying next.",
                task_id, slot_name, attempt["reasons"],
            )

    # ── Task-level metric ───────────────────────────────────────────────────
    router_tasks_total.labels(success="true" if success else "false").inc()

    logger.info(
        "task_id=%s finished success=%s final_model=%s total_cost=%d attempts=%d",
        task_id, success, final_model, total_cost, len(attempts),
    )

    return {
        "success": success,
        "final_model": final_model,
        "final_text": final_text,
        "total_virtual_cost_units": total_cost,
        "attempts": attempts,
    }
