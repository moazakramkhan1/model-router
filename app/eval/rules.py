"""
eval/rules.py – rule-based text evaluation (always free, runs on every attempt).
"""
import json
import re
from typing import Dict, Any, List, Tuple


# ---------------------------------------------------------------------------
# Individual rule functions – return (passed: bool, rule_name: str)
# ---------------------------------------------------------------------------

def rule_not_empty(text: str) -> Tuple[bool, str]:
    """The response must contain at least one non-whitespace character."""
    return bool(text and text.strip()), "response_not_empty"


def rule_no_error_markers(text: str) -> Tuple[bool, str]:
    """Response must not start with or prominently contain error/exception text."""
    lower = text.lower()
    bad_phrases = [
        "error:", "exception:", "traceback (most recent call",
        "http error", "500 internal server error",
        "connectionerror", "timeouterror",
    ]
    for phrase in bad_phrases:
        if phrase in lower:
            return False, f"no_error_markers:found={phrase!r}"
    return True, "no_error_markers"


def rule_valid_json(text: str) -> Tuple[bool, str]:
    """Response must be valid JSON (or contain a JSON code-block)."""
    stripped = text.strip()

    # Allow JSON wrapped in a markdown code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        json.loads(stripped)
        return True, "valid_json"
    except json.JSONDecodeError as exc:
        short = str(exc)[:100]
        return False, f"invalid_json:{short}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate(text: str, require_json: bool = False) -> Dict[str, Any]:
    """
    Run all applicable rules and return an evaluation dict.

    Returns
    -------
    {
        ok      : bool          – True iff ALL rules passed
        score   : float         – fraction of rules that passed (0.0 – 1.0)
        reasons : List[str]     – rule names (passing and failing)
        details : List[dict]    – per-rule { rule, ok } records
    }
    """
    checks: List[Tuple[bool, str]] = [
        rule_not_empty(text),
        rule_no_error_markers(text),
    ]
    if require_json:
        checks.append(rule_valid_json(text))

    details = [{"rule": name, "ok": ok} for ok, name in checks]
    passed = [ok for ok, _ in checks]
    score = round(sum(1 for p in passed if p) / len(passed), 2) if passed else 0.0
    overall = all(passed)

    return {
        "ok": overall,
        "score": score,
        "reasons": [name for _, name in checks],
        "details": details,
    }
