"""
router.py – core routing and cost-optimisation logic.

Flow for a single task
----------------------
1. Load models + policy configs.
2. Select the model path based on risk level.
3. For each slot in the path (up to max_attempts, within budget):
   a. Instantiate provider.
   b. Call provider.generate().
   c. Run judge on the result.
   d. Record attempt + metrics.
   e. If ok → break and return success.
4. Return final result regardless of outcome.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from config_loader import load_models, load_policy
from eval import judge
from metrics import (
    router_attempts_total,
    router_latency_ms,
    router_tasks_total,
    router_virtual_cost_units_total,
)
from providers import get_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Difficulty heuristic
# ---------------------------------------------------------------------------

_HARD_KEYWORDS = {
    "analyze", "analyse", "compare", "explain", "summarize", "summarise",
    "evaluate", "design", "architect", "optimize", "optimise", "debug",
    "reason", "write code", "implement", "generate code", "translate",
    "refactor", "critique", "assess",
}


def _estimate_difficulty(user_input: str) -> str:
    """Return 'hard' if prompt contains complexity keywords, else 'easy'."""
    lower = user_input.lower()
    for kw in _HARD_KEYWORDS:
        if kw in lower:
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
