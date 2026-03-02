"""
eval/judge.py – orchestrates rules-based + optional tiny LLM judge.

When USE_LLM_JUDGE=true (default), the rules judge runs first for a fast
score, then the tiny local LLM scores the response. If the LLM is
unavailable the rules score is used as the final verdict.

Set USE_LLM_JUDGE=false to skip the LLM judge entirely (e.g. in tests).
"""
import os
from typing import Any, Dict

from .rules import evaluate as rules_evaluate

_USE_LLM_JUDGE: bool = os.environ.get("USE_LLM_JUDGE", "true").lower() == "true"


async def judge(
    text: str,
    require_json: bool = False,
    user_input: str = "",
) -> Dict[str, Any]:
    """
    Evaluate a model response and return a consolidated verdict.

    Parameters
    ----------
    text         : raw string returned by the provider
    require_json : when True, the text must be parseable JSON
    user_input   : original prompt (used by the LLM judge for context)

    Returns
    -------
    {
        ok      : bool
        score   : float  (0.0 – 1.0)
        reasons : List[str]
    }
    """
    rules_result = rules_evaluate(text, require_json=require_json)

    # Short-circuit: if rules already fail hard (empty / error markers) and
    # LLM judge is disabled, return immediately.
    if not _USE_LLM_JUDGE or not text.strip():
        return {
            "ok": rules_result["ok"],
            "score": rules_result["score"],
            "reasons": rules_result["reasons"],
        }

    # LLM judge – uses same local model server, falls back gracefully
    from .llm_judge import llm_judge  # lazy import to avoid circular deps

    llm_result = await llm_judge(
        user_input=user_input,
        response_text=text,
        rules_score=rules_result["score"],
    )
    return {
        "ok": llm_result["ok"],
        "score": llm_result["score"],
        "reasons": llm_result["reasons"],
    }
