"""
eval/llm_judge.py – Tiny LLM judge running inside the server stack.

Uses the same local model server (Ollama) that serves the local_small slot.
No extra GPU, no extra container, no torch dependency in the router image.

The judge sends a structured scoring prompt to the local model and expects
a JSON response.  Falls back to the rules-based score if:
  - Ollama is unreachable
  - The model returns non-JSON output
  - The call exceeds the per-judge timeout

Environment variables
---------------------
LOCAL_BASE_URL  : base URL of the local model server (default http://ollama:11434)
LOCAL_MODEL     : model tag for the judge (same as the routing model)
LLM_JUDGE_TIMEOUT_S : per-call timeout in seconds (default 25)
"""
import json
import logging
import os
import re
from typing import Any, Dict

import httpx

log = logging.getLogger(__name__)

_JUDGE_BASE_URL: str = os.environ.get("LOCAL_BASE_URL", "http://ollama:11434")
_JUDGE_MODEL: str = os.environ.get(
    "LOCAL_MODEL", "hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M"
)
_JUDGE_TIMEOUT: float = float(os.environ.get("LLM_JUDGE_TIMEOUT_S", "25"))

_JUDGE_PROMPT = """\
You are a strict, impartial response quality judge.

USER PROMPT:
{user_input}

AI RESPONSE:
{response_text}

Score the AI response on a 0.0–1.0 scale for each criterion:
- relevance    : does it directly answer the prompt?
- coherence    : is it well-structured and grammatically readable?
- completeness : does it cover the topic sufficiently?

Compute overall = average of the three scores.

Reply ONLY with valid JSON, no extra text, exactly this schema:
{{"relevance": <float>, "coherence": <float>, "completeness": <float>, \
"overall": <float>, "reason": "<one sentence summary>"}}"""


async def llm_judge(
    user_input: str,
    response_text: str,
    rules_score: float,
) -> Dict[str, Any]:
    """
    Ask the tiny local LLM to score *response_text*.

    Returns
    -------
    dict with keys: ok (bool), score (float), reasons (list[str]), source (str)
    """
    prompt = _JUDGE_PROMPT.format(
        user_input=user_input[:600].strip(),
        response_text=response_text[:1200].strip(),
    )
    url = f"{_JUDGE_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": _JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 180,
            "temperature": 0.0,  # deterministic scoring
        },
    }

    try:
        async with httpx.AsyncClient(timeout=_JUDGE_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            raw_text: str = resp.json().get("response", "")

        # Extract the first JSON object from the response
        json_match = re.search(r"\{[\s\S]+?\}", raw_text)
        if not json_match:
            raise ValueError(f"No JSON object in judge output: {raw_text[:120]!r}")

        scores: dict = json.loads(json_match.group())
        overall: float = float(scores.get("overall", rules_score))
        # Clamp to [0.0, 1.0]
        overall = max(0.0, min(1.0, overall))
        reason: str = str(scores.get("reason", "llm_judge_scored"))

        return {
            "ok": overall >= 0.5,
            "score": round(overall, 3),
            "reasons": [f"llm_judge:{reason}"],
            "source": "llm",
        }

    except Exception as exc:
        log.warning(
            "LLM judge unavailable (%s: %s); falling back to rules score=%.2f.",
            type(exc).__name__,
            exc,
            rules_score,
        )
        return {
            "ok": rules_score >= 0.6,
            "score": round(rules_score, 3),
            "reasons": [f"llm_judge_fallback:{type(exc).__name__}"],
            "source": "rules_fallback",
        }
