"""
eval/judge.py – orchestrates all evaluation steps for a single model response.

Currently only the rules-based judge is implemented (cost: free).
A future LLM-judge can be wired in here without changing the router.
"""
from typing import Any, Dict

from .rules import evaluate as rules_evaluate


def judge(text: str, require_json: bool = False) -> Dict[str, Any]:
    """
    Evaluate a model response and return a consolidated verdict.

    Parameters
    ----------
    text         : raw string returned by the provider
    require_json : when True, the text must be parseable JSON

    Returns
    -------
    {
        ok      : bool
        score   : float  (0.0 – 1.0)
        reasons : List[str]
    }
    """
    result = rules_evaluate(text, require_json=require_json)
    return {
        "ok": result["ok"],
        "score": result["score"],
        "reasons": result["reasons"],
    }
